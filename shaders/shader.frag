#version 460

layout(location=0) in vec3 f_position; // сюда приходит из растеризатора интерполированная(промежут. значения на основе известных, в графике - заполнение между пикселями)
                                       // позиция пикселя в мировых координатах 
layout(location=1) in vec3 f_normal; // сюда приходит интерполированная нормаль
layout(location=2) in vec2 f_uv;     // сюда приходят интерполированные UV-координаты от вершинного шейдера

// LAB 4: Получаем позицию в пространстве света
layout(location=3) in vec4 f_light_space_pos;

layout(set=0, binding=1) uniform ModelUniforms {
    mat4 model;
    vec3 albedo_color; // Этот цвет будет умножаться на цвет из текстуры (тинтирование)
    float shininess;
    vec3 specular_color; // Этот цвет будет управлять цветом блика
    float _pad;
} model;


// Привязываем текстуры, которые были настроены в main.cpp для каждой модели
// binding=3 - основная текстура (альбедо/диффузная)
// binding=4 - текстура бликов (specular)
// binding=5 - текстура свечения (emissive)
layout(set=0, binding=3) uniform sampler2D albedo_sampler;
layout(set=0, binding=4) uniform sampler2D specular_sampler;
layout(set=0, binding=5) uniform sampler2D emissive_sampler;

// LAB 4: Новая текстура - карта теней.
// sampler2DShadow - специальный тип для карт теней со сравнением.
layout(set=0, binding=6) uniform sampler2DShadow shadow_map_sampler;

struct PointLight {
    vec3 position;
    vec3 color;
    float constant;
    float linear;
    float quadratic;
    float _pad;
};

// объявляем SSBO, привязанный к слоту 2
layout(set=0, binding=2) readonly buffer LightSSBO {
    PointLight point_lights[8];
    uint point_light_count;
    vec3 _pad[3];
} lights;


// Добавили поле time для анимации нетривиального сэмплирования
layout(push_constant) uniform PushConstants {
    vec3 camera_position;
    float time; // Время для анимации в доп. задании
    vec3 ambient_color;
    float _p1;
    vec3 directional_dir;
    float _p2;
    vec3 directional_color;
    float _p3;

    vec3 spot_pos;
    float _s_p0;
    vec3 spot_dir;
    float _s_p1;
    vec3 spot_col;
    float _s_p2;
    float inner_cutOff_cos; 
    float outer_cutOff_cos; 

    float spot_constant;
    float spot_linear;
    float spot_quadratic;

    float _s_p3;
    float _s_p4;
} pc;

layout(location=0) out vec4 final_color;


// LAB 4: Функция для расчета тени
float calculate_shadow(vec4 light_space_pos) {
    // 1. Перспективное деление
    vec3 proj_coords = light_space_pos.xyz / light_space_pos.w;
    // 2. Перевод в [0, 1]
    proj_coords = proj_coords * 0.5 + 0.5;

    // Если за пределами карты — света нет (или есть, зависит от логики, тут считаем что свет есть)
    if(proj_coords.z > 1.0) return 1.0;

    float current_depth = proj_coords.z;

    // --- НАСТРОЙКА BIAS (Убирает полоски) ---
    // Увеличили bias, так как у тебя большие координаты сцены.
    // Это убьет полоски на полу наповал.
    float bias = 0.005; 

    // --- PCF (МЯГКИЕ ТЕНИ) ---
    float shadow = 0.0;
    // Узнаем размер одного пикселя текстуры (чтобы знать, насколько шагать)
    // shadow_map_sampler - это sampler2DShadow, textureSize возвращает размер.
    vec2 texelSize = 1.0 / textureSize(shadow_map_sampler, 0);

    // Проходимся циклом 3x3 вокруг пикселя
    for(int x = -1; x <= 1; ++x) {
        for(int y = -1; y <= 1; ++y) {
            // Берем соседние пиксели
            float pcfDepth = texture(shadow_map_sampler, vec3(proj_coords.xy + vec2(x, y) * texelSize, current_depth - bias)); 
            shadow += pcfDepth;
        }
    }
    // Усредняем (делим на 9 выборок)
    shadow /= 9.0;
    
    return shadow;
}





// Модель освещения Блинн-Фонга
// N - нормаль, V - вектор К камере, L - вектор К свету
// LAB 3: Функция модифицирована для приема цвета альбедо и спекуляра в качестве параметров
vec3 calculate_blinn_phong(vec3 N, vec3 V, vec3 L, vec3 light_color, float attenuation, vec3 albedo, vec3 specular) {
    float diff = max(dot(N, L), 0.0); // диффузный компонент (имитация света на матовых поверхностях)
    vec3 H = normalize(V + L); // вычисляем вектор полупути
    float spec = pow(max(dot(N, H), 0.0), model.shininess); // вычисляем блик (имитация света на глянцевых поверхностях)
    
    // Вместо глобальных model.albedo_color/specular_color используем переданные значения из текстур
    return light_color * attenuation * (albedo * diff + specular * spec);
}

// делаем прожекторный свет; помимо вызова Блинна-Фонга, проверяем, попадает ли пиксель в конус света, вычисляем
// интенсивность с учетом плавных краев
vec3 calculate_spot_light(vec3 N, vec3 V, vec3 albedo, vec3 specular) {
    vec3 light_vec = pc.spot_pos - f_position;
    float dist = length(light_vec);
    
    if (dist == 0.0) return vec3(0.0);
    
    
    vec3 L = light_vec / dist;
    
    vec3 D = pc.spot_dir; 
    
    float theta = dot(-L, D);
    
    // float constant = 1.0;
    // float linear = 0.14; 
    // float quadratic = 0.07;

    float constant = pc.spot_constant;
    float linear = pc.spot_linear;
    float quadratic = pc.spot_quadratic;
    float denominator = constant + linear * dist + quadratic * dist * dist;
    float attenuation = 1.0 / max(denominator, 0.00001);
    
    float inner_cos = pc.inner_cutOff_cos;
    float outer_cos = pc.outer_cutOff_cos;
    float epsilon = inner_cos - outer_cos; 
    
    float intensity = 0.0;
    
    if (theta > outer_cos) {
        if (epsilon > 0.0001) {
            intensity = clamp((theta - outer_cos) / epsilon, 0.0, 1.0);
        } else {
            intensity = (theta >= inner_cos) ? 1.0 : 0.0;
        }
    }

    return calculate_blinn_phong(N, V, L, pc.spot_col, attenuation * intensity, albedo, specular);
}

// логика расчета цвета на основе модели Блинн-Фонга
// разные типы источников света: ambient - рассеянный; directional - направленный; point - точечные; spot - прожектор
void main() {
    // Искажаем входящие текстурные координаты f_uv с помощью синуса и косинуса,
    // используя время (pc.time), чтобы создать анимированный эффект "волн" или "искажения".
    vec2 distorted_uv = f_uv;
    float distortion_strength = 0.05; // Сила искажения
    float distortion_speed = 2.0;    // Скорость анимации
    float distortion_frequency = 10.0; // Частота (густота) волн
    distorted_uv.x += sin(f_uv.y * distortion_frequency + pc.time * distortion_speed) * distortion_strength;
    distorted_uv.y += cos(f_uv.x * distortion_frequency + pc.time * distortion_speed) * distortion_strength;

    // Используем (искаженные) координаты для чтения цвета из текстур.
    // .rgb отбрасывает альфа-канал, оставляя только цвет.
    // через функцию texture мы берем цвет из текстуры
    vec3 albedo = texture(albedo_sampler, distorted_uv).rgb * model.albedo_color;
    
    // Сэмплируем specular-карту. Белые участки будут сильно бликовать, черные - нет.
    // Specular-карта используется для модуляции силы блика
    // Значение цвета из specular_sampler (обычно от 0 до 1) умножается на цвет блика в формуле освещения. 
    // Если в specular-текстуре пиксель черный (0), блика не будет. Если белый (1) — блик будет максимальной силы.
    vec3 specular = texture(specular_sampler, distorted_uv).rgb * model.specular_color;

    vec3 N = normalize(f_normal);
    vec3 V = normalize(pc.camera_position - f_position); // вычисляем вектор от точки к камере
    
    // Начинаем расчет освещения с компонента ambient, используя цвет из albedo-текстуры
    vec3 color = pc.ambient_color * albedo; // добавляем фоновый цвет; ambient - умножение цвета окружения на цвет объекта

    // LAB 4: Вычисляем фактор тени
    float shadow = calculate_shadow(f_light_space_pos);
    // Добавляем проверку, чтобы избежать теней "под" плоскостью
    
    
    //if (dot(N, normalize(-pc.directional_dir)) < 0.1) {
    //    shadow = 1.0;
    //}



    vec3 dir_light_dir = normalize(-pc.directional_dir);
    color += shadow * calculate_blinn_phong(N, V, dir_light_dir, pc.directional_color, 1.0, albedo, specular); // добавляем направленный цвет; постоянное направление цвета

    // Point lights
    // добавляем точечные цвета
    // для каждого источника вычисляется направление L, затухание по закону обратных квадратов
    for (uint i = 0; i < lights.point_light_count; ++i) {
        vec3 light_vec = lights.point_lights[i].position - f_position;
        float dist = length(light_vec);
        vec3 L = normalize(light_vec);
        float denominator = lights.point_lights[i].constant +
                            lights.point_lights[i].linear * dist +
                            lights.point_lights[i].quadratic * dist * dist;
        float attenuation = 1.0 / (0.25 + denominator);

        color += calculate_blinn_phong(N, V, L, lights.point_lights[i].color, attenuation, albedo, specular);
    }

    color += calculate_spot_light(N, V, albedo, specular); // добавляем прожекторный цвет
    
    // Сэмплируем карту свечения и просто добавляем её цвет к финальному результату.
    // Это заставит участки объекта светиться независимо от освещения.
    vec3 emissive = texture(emissive_sampler, distorted_uv).rgb;
    color += emissive;

    final_color = vec4(color, 1.0); // финальный цвет, который будет записан в пиксель на экране
}