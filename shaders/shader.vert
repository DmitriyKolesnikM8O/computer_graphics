#version 450

layout (location = 0) in vec3 v_position; // сюда GPU подает позицию одной вершины из буфера
layout (location = 1) in vec3 v_normal;
layout (location = 2) in vec2 v_uv; // получаем uv из буфера вершин

layout (location = 0) out vec3 f_position;
layout (location = 1) out vec3 f_normal;
layout (location = 2) out vec2 f_uv; // объявляем канал для передачи дальше

// LAB 4: Новое поле - позиция вершины в пространстве света
layout (location = 3) out vec4 f_light_space_pos;



// Данные, одинаковые для всех вершин в рамках одного объекта.
// model Матрица переводит вершину из локальных координат в мировые
// view_projection - из мировых в пространство экрана
layout (binding = 0, std140) uniform SceneUniforms {
	mat4 view_projection;
};
layout (binding = 1, std140) uniform ModelUniforms {
	mat4 model;
	vec3 albedo_color;
	float shininess;
	vec3 specular_color;
	float _pad;
};

// Push-константы теперь нужны и в вершинном шейдере
// чтобы получить матрицу света.
// В C++ мы передаем всю структуру Push, но здесь нам нужна только матрица,
// поэтому мы указываем ее смещение в байтах.
layout(push_constant) uniform PushConstants {
    layout(offset = 96) mat4 light_space_matrix;
} pc;

void main() {
	vec4 position_world = model * vec4(v_position, 1.0f);
	vec4 normal_world = model * vec4(v_normal, 0.0f);

	gl_Position = view_projection * position_world; // результат работы шейдера, вычисляется финал. позиция вершины

	// также передаем позицию и нормаль в мировых координатах дальше конвейеру (Rendering Pipeline)
	f_position = position_world.xyz;	
	f_normal = normal_world.xyz;
	f_uv = v_uv;

	// LAB 4: Вычисляем позицию в пространстве света и передаем ее дальше
    f_light_space_pos = pc.light_space_matrix * position_world;
}