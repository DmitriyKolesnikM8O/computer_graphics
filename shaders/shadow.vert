#version 450

layout(location = 0) in vec3 v_position;

// УБРАЛИ SceneUniforms (binding = 0), ТАК КАК ОН КОНФЛИКТУЕТ С КАМЕРОЙ
// Вместо этого принимаем матрицу через Push Constant
// offset = 96, так как в твоем C++ коде (main.cpp) для shadow pipeline 
// ты указал offset = 96.
layout(push_constant) uniform Push {
    layout(offset = 0) mat4 light_space_matrix;
} push;

layout(binding = 1, std140) uniform ModelUniforms {
    mat4 model;
};

void main() {
    // Используем push.light_space_matrix вместо UBO
    gl_Position = push.light_space_matrix * model * vec4(v_position, 1.0f);
}