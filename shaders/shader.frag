#version 450

layout (location = 0) in vec3 f_position;
layout (location = 1) in vec3 f_normal;
layout (location = 2) in vec2 f_uv;
layout (location = 3) in vec4 f_pos_light_space;
layout (location = 4) in vec4 f_pos_spot_light_space[2];

layout (location = 0) out vec4 final_color;

struct PointLight {
    vec3 position;
    float intensity;
    vec3 color;
    float _pad0;
};

struct SpotLight {
    vec3 position;
    float radius;
    vec3 direction;
    float angle;
    vec3 color;
    float _pad0;
};

layout (set = 0, binding = 0, std140) uniform SceneUniforms {
    mat4 view_projection;
    mat4 light_view_projection;
    vec3 camera_pos;
    float _pad0;
    
    vec3 ambient_color;
    float _pad1;
    vec3 ambient_light_intensity;
    float _pad2;
    
    vec3 sun_light_direction;
    float _pad3;
    vec3 sun_light_color;
    float _pad4;

    uint point_light_count;
    uint spot_light_count;
    uint shadow_casting_spot_count;
    float _pad5;
    
    mat4 spot_light_matrices[2];
};

layout (set = 0, binding = 1, std140) uniform ModelUniforms {
    mat4 model;
    vec3 albedo_color;
    float _pad6;
    vec3 specular_color;
    float _pad7;
    float shininess;
};

layout (set = 0, binding = 2, std430) readonly buffer PointLightsBuffer {
    PointLight point_lights[];
};

layout (set = 0, binding = 3, std430) readonly buffer SpotLightsBuffer {
    SpotLight spot_lights[];
};

layout (set = 1, binding = 0) uniform sampler2D texSampler;

layout (set = 2, binding = 0) uniform sampler2DShadow shadowMap; //тип сэмплера для теней

// массив сэмплеров для прожектора

layout (set = 2, binding = 1) uniform sampler2DShadow spotShadowMap0;
layout (set = 2, binding = 2) uniform sampler2DShadow spotShadowMap1;

// тут все тени рассчитываем
float calculateShadow(vec4 lightSpacePos, vec3 normal, vec3 lightDir, sampler2DShadow shadowSampler) {
    
    // Превращаем однородные координаты (4D) в обычные 3D
    vec3 projCoords = lightSpacePos.xyz / lightSpacePos.w;

    // Переводим из диапазона [-1, 1] в [0, 1] (как хранятся текстуры)
    projCoords.xy = projCoords.xy * 0.5 + 0.5;

    // Если за границами карты теней — света нет (или есть, зависит от логики, тут возвращаем 1.0)
    if (projCoords.z > 1.0 || projCoords.x < 0.0 || projCoords.x > 1.0 || 
        projCoords.y < 0.0 || projCoords.y > 1.0) {
        return 1.0;
    }

    // эта штука должна помогать бороться с теневой рябью
    // пытаемся сдвинуть глубину сравнения чуть назад
    // меняется в зависимости от угла наклона поверхности к свету
    float bias = max(0.005 * (1.0 - dot(normal, lightDir)), 0.0005);

    // 3 компонент для сравнения PCF (происходит автоматически внутри texture)
    // biast тут для борьбы с артефактами
    // shadowSampler — это sampler2DShadow.
    // Функция texture() принимает vec3, где:
    // .xy — координаты на карте
    // .z  — глубина нашего пикселя (минус bias)
    // Видеокарта САМА сравнивает .z со значением в текстуре,
    // усредняет 4 соседних пикселя и возвращает результат (0.0..1.0).
    float shadow = texture(shadowSampler, vec3(projCoords.xy, projCoords.z - bias));

    return shadow; // 1.0 свет или 0.0 тень с градацией по краям
}


// Считает диффузную и спекулярную составляющие
vec3 calculateBlinnPhong(vec3 lightDir, vec3 lightColor, vec3 normal, vec3 viewDir, vec3 albedo) {
    // Diffuse
    float diff = max(dot(normal, lightDir), 0.0);
    
    // Оптимизация: если свет светит в "спину", нет смысла считать блики
    if (diff == 0.0) return vec3(0.0);

    vec3 diffuse = albedo * lightColor * diff;

    // Specular (Blinn-Phong uses Half-Vector)
    vec3 halfDir = normalize(lightDir + viewDir);
    float spec = pow(max(dot(normal, halfDir), 0.0), shininess);
    vec3 specular = specular_color * lightColor * spec;

    // Чем больше угол между взглядом и нормалью (край объекта), тем ярче.
    // 1.0 - dot(N, V) дает "ободок".
    float rimFactor = 1.0 - max(dot(normal, viewDir), 0.0);
    // Возводим в степень (напр. 3 или 4), чтобы ободок был тонким.
    rimFactor = pow(rimFactor, 4.0);
    // Делаем его слабым (0.3) и цвета света
    vec3 rim = lightColor * rimFactor * 0.3 * albedo;

    return diffuse + specular;
}

void main() {
    vec3 normal = normalize(f_normal);
    vec3 view_dir = normalize(camera_pos - f_position);

    vec4 texColor = texture(texSampler, f_uv);
    vec3 albedoWithTexture = albedo_color * texColor.rgb;
    
    // 1. Ambient (Фоновое освещение)
    vec3 color = ambient_light_intensity * albedoWithTexture;

    // 2. Sun (Directional Light)
    vec3 sun_dir = normalize(-sun_light_direction);
    
    // Считаем тень от солнца
    float shadowFactor = calculateShadow(f_pos_light_space, normal, sun_dir, shadowMap);
    
    // Считаем свет и применяем тень
    color += calculateBlinnPhong(sun_dir, sun_light_color, normal, view_dir, albedoWithTexture) * shadowFactor;

    // 3. Point Lights (Точечные)
    for (uint i = 0; i < point_light_count; ++i) {
        PointLight light = point_lights[i];
        vec3 light_vec = light.position - f_position;
        float distance = length(light_vec);
        vec3 ldir = normalize(light_vec);

        // Затухание
        float light_falloff = light.intensity / (distance * distance + 0.0001);
        
        color += calculateBlinnPhong(ldir, light.color, normal, view_dir, albedoWithTexture) * light_falloff;
    }

    // 4. Spot Lights (Прожекторы)
    for (uint i = 0; i < spot_light_count; ++i) {
        SpotLight light = spot_lights[i];
        vec3 light_vec = light.position - f_position;
        float distance = length(light_vec);

        if (distance > light.radius) continue;

        vec3 ldir = normalize(light_vec);
        vec3 spot_dir = normalize(light.direction);
        
        // Проверка угла конуса
        float theta = dot(-ldir, spot_dir);
        float cutoff_cos = cos(light.angle);

        if (theta > cutoff_cos) {
            // Мягкие края прожектора
            float distance_attenuation = clamp(1.0 - (distance / light.radius), 0.0, 1.0);
            float outer_cutoff = cos(light.angle);
            float inner_cutoff = cos(light.angle * 0.8);
            float epsilon = inner_cutoff - outer_cutoff;
            float spot_intensity = clamp((theta - outer_cutoff) / epsilon, 0.0, 1.0);
            
            float total_attenuation = distance_attenuation * spot_intensity;

            // Тени от прожекторов
            float spotShadow = 1.0;
            if (i < shadow_casting_spot_count) {
                // выбираем нужную карту в зависимости от индекса источника
                if (i == 0) {
                    spotShadow = calculateShadow(f_pos_spot_light_space[0], normal, ldir, spotShadowMap0);
                } else if (i == 1) {
                    spotShadow = calculateShadow(f_pos_spot_light_space[1], normal, ldir, spotShadowMap1);
                }
            }

            // применяем тень к свету прожектора
            color += calculateBlinnPhong(ldir, light.color, normal, view_dir, albedoWithTexture) * total_attenuation * spotShadow;
        }
    }

    final_color = vec4(color, texColor.a);
}