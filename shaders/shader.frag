#version 460

layout(location=0) in vec3 f_position;
layout(location=1) in vec3 f_normal;
layout(location=2) in vec2 f_uv;

layout(set=0, binding=1) uniform ModelUniforms {
    mat4 model;
    vec3 albedo_color;
    float shininess;
    vec3 specular_color;
    float _pad;
} model;

struct PointLight {
    vec3 position;
    vec3 color;
    float constant;
    float linear;
    float quadratic;
    float _pad;
};

layout(set=0, binding=2) buffer LightSSBO {
    PointLight point_lights[8];
    uint point_light_count;
    vec3 _pad[3];
} lights;

layout(push_constant) uniform PushConstants {
    vec3 camera_position;
    float _p0;
    vec3 ambient_color;
    float _p1;
    vec3 directional_dir;
    float _p2;
    vec3 directional_color;
    float _p3;
} pc;

layout(location=0) out vec4 final_color;

vec3 calculate_blinn_phong(vec3 N, vec3 V, vec3 L, vec3 light_color, float attenuation) {
    float diff = max(dot(N, L), 0.0);
    vec3 H = normalize(V + L);
    float spec = pow(max(dot(N, H), 0.0), model.shininess);
    return light_color * attenuation * (model.albedo_color * diff + model.specular_color * spec);
}

void main() {
    vec3 N = normalize(f_normal);
    vec3 V = normalize(pc.camera_position - f_position);
    vec3 color = pc.ambient_color * model.albedo_color;

    // Directional light
    vec3 dir_light_dir = normalize(-pc.directional_dir);
    color += calculate_blinn_phong(N, V, dir_light_dir, pc.directional_color, 1.0);

    // Point lights
    for (uint i = 0; i < lights.point_light_count; ++i) {
        vec3 light_vec = lights.point_lights[i].position - f_position;
        float dist = length(light_vec);
        if (dist > 0.001) {
            vec3 L = light_vec / dist;
            float attenuation = 1.0 / (
                lights.point_lights[i].constant +
                lights.point_lights[i].linear * dist +
                lights.point_lights[i].quadratic * dist * dist
            );
            color += calculate_blinn_phong(N, V, L, lights.point_lights[i].color, attenuation);
        }
    }

    final_color = vec4(color, 1.0);
}