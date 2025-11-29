#version 450

layout(location = 0) in vec3 v_position;

// ЧИТАЕМ МАТРИЦУ ИЗ UBO, А НЕ ИЗ PUSH-КОНСТАНТЫ
layout(binding = 0, std140) uniform SceneUniforms {
	mat4 light_space_matrix;
};

layout(binding = 1, std140) uniform ModelUniforms {
	mat4 model;
};

void main() {
	gl_Position = light_space_matrix * model * vec4(v_position, 1.0f);
}