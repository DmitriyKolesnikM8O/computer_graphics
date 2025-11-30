#version 450

layout(location = 0) in vec3 inPosition;

layout(push_constant) uniform Constants {
    mat4 model;
    mat4 lightViewProj;
} push;

void main() {
    // Мы берем позицию вершины (inPosition)
    // Умножаем на матрицу модели (ставим в мир)
    // Умножаем на матрицу света (lightViewProj)
    // И записываем результат в gl_Position
    gl_Position = push.lightViewProj * push.model * vec4(inPosition, 1.0);
}
