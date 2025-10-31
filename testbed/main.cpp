#include "veekay/input.hpp"
#include <cstdint>
#include <climits>
#include <vector>
#include <iostream>
#include <fstream>
#include <algorithm>

#define _USE_MATH_DEFINES
#include <math.h>
#include <cmath>
#include <veekay/veekay.hpp>

#include <imgui.h>
#include <vulkan/vulkan_core.h>
#include <lodepng.h>

constexpr float PI = 3.14159265359f;
inline float to_radians(float degrees) { return degrees * (PI / 180.0f); }
inline float to_degrees(float radians) { return radians * (180.0f / PI); }

namespace {

constexpr uint32_t max_models = 1024;
constexpr uint32_t max_point_lights = 8;

struct Vertex {
	veekay::vec3 position;
	veekay::vec3 normal;
	veekay::vec2 uv;
};

struct SceneUniforms {
	veekay::mat4 view_projection;
};

struct Material {
	veekay::vec3 albedo = {1.0f, 1.0f, 1.0f};
	float _pad0;
	veekay::vec3 specular = {1.0f, 1.0f, 1.0f};
	float shininess = 32.0f;
};

struct ModelUniforms {
	veekay::mat4 model;
	veekay::vec3 albedo_color;
	float shininess;
	veekay::vec3 specular_color;
	float _pad;
};

struct Mesh {
	veekay::graphics::Buffer* vertex_buffer;
	veekay::graphics::Buffer* index_buffer;
	uint32_t indices;
};

struct Transform {
	veekay::vec3 position = {};
	veekay::vec3 scale = {1.0f, 1.0f, 1.0f};
	veekay::vec3 rotation = {};

	veekay::mat4 matrix() const;
};

struct Model {
	Mesh mesh;
	Transform transform;
	Material material;
};

struct Camera {
	constexpr static float default_fov = 60.0f;
	constexpr static float default_near_plane = 0.01f;
	constexpr static float default_far_plane = 100.0f;

	veekay::vec3 position = {};
	veekay::vec3 rotation = {};

	float fov = default_fov;
	float near_plane = default_near_plane;
	float far_plane = default_far_plane;

	veekay::mat4 view() const;
	veekay::mat4 view_projection(float aspect_ratio) const;


    veekay::vec3 target = {0.0f, -0.5f, 0.0f};
    bool is_look_at = false;
};

struct AmbientLight {
	veekay::vec3 color = {0.1f, 0.1f, 0.1f};
};

struct DirectionalLight {
	veekay::vec3 direction = {0.0f, -1.0f, 0.0f};
	veekay::vec3 color = {1.0f, 1.0f, 1.0f};
};

struct PointLight {
	veekay::vec3 position = {0.0f, 1.0f, 0.0f};
    float position_pad;
	veekay::vec3 color = {1.0f, 1.0f, 1.0f};
    float color_pad;
	float constant = 1.0f;
	float linear = 0.09f;
	float quadratic = 0.032f;
	float _pad;
};

struct SpotLight {
	veekay::vec3 position = {0.0f, 2.0f, 0.0f};
	veekay::vec3 direction = {0.0f, -1.0f, 0.0f}; // Направление конуса
	veekay::vec3 color = {1.0f, 1.0f, 1.0f};
	float inner_cutOff = 12.5f; // Угол конуса (в градусах)
	float outer_cutOff = 17.5f;
};

struct LightSSBO {
	PointLight point_lights[max_point_lights];
	uint32_t point_light_count = 0;
	veekay::vec3 _pad[3];
};

inline namespace {
	Camera camera{
		.position = {0.0f, -0.5f, -3.0f}
	};

	std::vector<Model> models;

	AmbientLight ambient_light;
	DirectionalLight directional_light;
	LightSSBO light_ssbo;

    SpotLight spot_light;
}

inline namespace {
	VkShaderModule vertex_shader_module;
	VkShaderModule fragment_shader_module;

	VkDescriptorPool descriptor_pool;
	VkDescriptorSetLayout descriptor_set_layout;
	VkDescriptorSet descriptor_set;

	VkPipelineLayout pipeline_layout;
	VkPipeline pipeline;

	veekay::graphics::Buffer* scene_uniforms_buffer;
	veekay::graphics::Buffer* model_uniforms_buffer;
	veekay::graphics::Buffer* light_ssbo_buffer;

	Mesh plane_mesh;
	Mesh cube_mesh;

	veekay::graphics::Texture* missing_texture;
	VkSampler missing_texture_sampler;

	veekay::graphics::Texture* texture;
	VkSampler texture_sampler;
}

float toRadians(float degrees) {
	return degrees * float(M_PI) / 180.0f;
}

veekay::mat4 Transform::matrix() const {
	auto t = veekay::mat4::translation(position);
	auto s = veekay::mat4::scaling(scale);
	auto rx = veekay::mat4::rotation({1,0,0}, toRadians(rotation.x));
	auto ry = veekay::mat4::rotation({0,1,0}, toRadians(rotation.y));
	auto rz = veekay::mat4::rotation({0,0,1}, toRadians(rotation.z));
	return t * rz * ry * rx * s;
}

veekay::mat4 Camera::view() const {
    if (is_look_at) {
        
        return veekay::mat4::look_at(position, target, {0.0f, 1.0f, 0.0f});
    }



	auto t = veekay::mat4::translation(-position);
	auto rx = veekay::mat4::rotation({1,0,0}, toRadians(-rotation.x));
	auto ry = veekay::mat4::rotation({0,1,0}, toRadians(-rotation.y));
	auto rz = veekay::mat4::rotation({0,0,1}, toRadians(-rotation.z));
	return rz * ry * rx * t;
}

veekay::mat4 Camera::view_projection(float aspect_ratio) const {
	auto projection = veekay::mat4::projection(fov, aspect_ratio, near_plane, far_plane);
	return view() * projection;
}

VkShaderModule loadShaderModule(const char* path) {
	std::ifstream file(path, std::ios::binary | std::ios::ate);
	if (!file.is_open()) return nullptr;
	size_t size = file.tellg();
	std::vector<uint32_t> buffer(size / sizeof(uint32_t));
	file.seekg(0);
	file.read(reinterpret_cast<char*>(buffer.data()), size);
	file.close();

	VkShaderModuleCreateInfo info{
		.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
		.codeSize = size,
		.pCode = buffer.data(),
	};

	VkShaderModule result;
	if (vkCreateShaderModule(veekay::app.vk_device, &info, nullptr, &result) != VK_SUCCESS) {
		return nullptr;
	}
	return result;
}

void initialize(VkCommandBuffer cmd) {
	VkDevice& device = veekay::app.vk_device;

	vertex_shader_module = loadShaderModule("./shaders/shader.vert.spv");
	if (!vertex_shader_module) {
		std::cerr << "Failed to load Vulkan vertex shader from file\n";
		veekay::app.running = false;
		return;
	}

	fragment_shader_module = loadShaderModule("./shaders/shader.frag.spv");
	if (!fragment_shader_module) {
		std::cerr << "Failed to load Vulkan fragment shader from file\n";
		veekay::app.running = false;
		return;
	}

	VkPipelineShaderStageCreateInfo stage_infos[2] = {
		{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.stage = VK_SHADER_STAGE_VERTEX_BIT,
			.module = vertex_shader_module,
			.pName = "main",
		},
		{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.stage = VK_SHADER_STAGE_FRAGMENT_BIT,
			.module = fragment_shader_module,
			.pName = "main",
		}
	};

	VkVertexInputBindingDescription buffer_binding{
		.binding = 0,
		.stride = sizeof(Vertex),
		.inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
	};

	VkVertexInputAttributeDescription attributes[] = {
		{ .location = 0, .binding = 0, .format = VK_FORMAT_R32G32B32_SFLOAT, .offset = offsetof(Vertex, position) },
		{ .location = 1, .binding = 0, .format = VK_FORMAT_R32G32B32_SFLOAT, .offset = offsetof(Vertex, normal) },
		{ .location = 2, .binding = 0, .format = VK_FORMAT_R32G32_SFLOAT, .offset = offsetof(Vertex, uv) },
	};

	VkPipelineVertexInputStateCreateInfo input_state_info{
		.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
		.vertexBindingDescriptionCount = 1,
		.pVertexBindingDescriptions = &buffer_binding,
		.vertexAttributeDescriptionCount = 3,
		.pVertexAttributeDescriptions = attributes,
	};

	VkPipelineInputAssemblyStateCreateInfo assembly_state_info{
		.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
		.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
	};

	VkPipelineRasterizationStateCreateInfo raster_info{
		.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
		.polygonMode = VK_POLYGON_MODE_FILL,
		.cullMode = VK_CULL_MODE_BACK_BIT,
		.frontFace = VK_FRONT_FACE_CLOCKWISE,
		.lineWidth = 1.0f,
	};

	VkPipelineMultisampleStateCreateInfo sample_info{
		.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
		.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
	};

	VkViewport viewport{
		.width = static_cast<float>(veekay::app.window_width),
		.height = static_cast<float>(veekay::app.window_height),
		.minDepth = 0.0f,
		.maxDepth = 1.0f,
	};

	VkRect2D scissor{
		.extent = {veekay::app.window_width, veekay::app.window_height},
	};

	VkPipelineViewportStateCreateInfo viewport_info{
		.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
		.viewportCount = 1,
		.pViewports = &viewport,
		.scissorCount = 1,
		.pScissors = &scissor,
	};

	VkPipelineDepthStencilStateCreateInfo depth_info{
		.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
		.depthTestEnable = true,
		.depthWriteEnable = true,
		.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL,
	};

	VkPipelineColorBlendAttachmentState attachment_info{
		.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
	};

	VkPipelineColorBlendStateCreateInfo blend_info{
		.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
		.logicOpEnable = false,
		.attachmentCount = 1,
		.pAttachments = &attachment_info
	};

	{
		VkDescriptorPoolSize pools[] = {
			{ VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 8 },
			{ VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 8 },
			{ VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 8 },
			{ VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 8 }
		};

		VkDescriptorPoolCreateInfo info{
			.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
			.maxSets = 1,
			.poolSizeCount = 4,
			.pPoolSizes = pools,
		};

		if (vkCreateDescriptorPool(device, &info, nullptr, &descriptor_pool) != VK_SUCCESS) {
			std::cerr << "Failed to create Vulkan descriptor pool\n";
			veekay::app.running = false;
			return;
		}
	}

	{
		VkDescriptorSetLayoutBinding bindings[] = {
			{ 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT },
			{ 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT },
			{ 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT }
		};

		VkDescriptorSetLayoutCreateInfo info{
			.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
			.bindingCount = 3,
			.pBindings = bindings,
		};

		if (vkCreateDescriptorSetLayout(device, &info, nullptr, &descriptor_set_layout) != VK_SUCCESS) {
			std::cerr << "Failed to create Vulkan descriptor set layout\n";
			veekay::app.running = false;
			return;
		}
	}

	{
		VkDescriptorSetAllocateInfo info{
			.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
			.descriptorPool = descriptor_pool,
			.descriptorSetCount = 1,
			.pSetLayouts = &descriptor_set_layout,
		};

		if (vkAllocateDescriptorSets(device, &info, &descriptor_set) != VK_SUCCESS) {
			std::cerr << "Failed to create Vulkan descriptor set\n";
			veekay::app.running = false;
			return;
		}
	}

	VkPushConstantRange push_range{
		.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
		.size = 128
	};

	VkPipelineLayoutCreateInfo layout_info{
		.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
		.setLayoutCount = 1,
		.pSetLayouts = &descriptor_set_layout,
		.pushConstantRangeCount = 1,
		.pPushConstantRanges = &push_range,
	};

	if (vkCreatePipelineLayout(device, &layout_info, nullptr, &pipeline_layout) != VK_SUCCESS) {
		std::cerr << "Failed to create Vulkan pipeline layout\n";
		veekay::app.running = false;
		return;
	}

	VkGraphicsPipelineCreateInfo info{
		.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
		.stageCount = 2,
		.pStages = stage_infos,
		.pVertexInputState = &input_state_info,
		.pInputAssemblyState = &assembly_state_info,
		.pViewportState = &viewport_info,
		.pRasterizationState = &raster_info,
		.pMultisampleState = &sample_info,
		.pDepthStencilState = &depth_info,
		.pColorBlendState = &blend_info,
		.layout = pipeline_layout,
		.renderPass = veekay::app.vk_render_pass,
	};

	if (vkCreateGraphicsPipelines(device, nullptr, 1, &info, nullptr, &pipeline) != VK_SUCCESS) {
		std::cerr << "Failed to create Vulkan pipeline\n";
		veekay::app.running = false;
		return;
	}

	scene_uniforms_buffer = new veekay::graphics::Buffer(
		sizeof(SceneUniforms), nullptr, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);

	model_uniforms_buffer = new veekay::graphics::Buffer(
		max_models * sizeof(ModelUniforms), nullptr, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);

	light_ssbo_buffer = new veekay::graphics::Buffer(
		sizeof(LightSSBO), nullptr, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    light_ssbo.point_lights[0] = { {2.0f, 1.0f, 0.0f}, 0.0f, {1.0f, 0.0f, 0.0f}, 0.0f, 1.0f, 0.14f, 0.07f };
    light_ssbo.point_lights[1] = { {-2.0f, 1.0f, 0.0f}, 0.0f, {0.0f, 1.0f, 0.0f}, 0.0f, 1.0f, 0.14f, 0.07f };
    light_ssbo.point_light_count = 2;
    *(LightSSBO*)light_ssbo_buffer->mapped_region = light_ssbo;
	{
		VkSamplerCreateInfo info{
			.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
			.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
		};

		if (vkCreateSampler(device, &info, nullptr, &missing_texture_sampler) != VK_SUCCESS) {
			std::cerr << "Failed to create Vulkan texture sampler\n";
			veekay::app.running = false;
			return;
		}

		uint32_t pixels[] = { 0xff000000, 0xffff00ff, 0xffff00ff, 0xff000000 };
		missing_texture = new veekay::graphics::Texture(cmd, 2, 2, VK_FORMAT_B8G8R8A8_UNORM, pixels);
	}

	{
		VkDescriptorBufferInfo buffer_infos[] = {
			{ scene_uniforms_buffer->buffer, 0, sizeof(SceneUniforms) },
			{ model_uniforms_buffer->buffer, 0, sizeof(ModelUniforms) },
			{ light_ssbo_buffer->buffer, 0, VK_WHOLE_SIZE }
		};

		VkWriteDescriptorSet write_infos[] = {
			{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, descriptor_set, 0, 0, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, nullptr, &buffer_infos[0] },
			{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, descriptor_set, 1, 0, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, nullptr, &buffer_infos[1] },
			{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, descriptor_set, 2, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &buffer_infos[2] }
		};

		vkUpdateDescriptorSets(device, 3, write_infos, 0, nullptr);
	}

	{
		std::vector<Vertex> vertices = {
			{{-5.0f, 0.0f, 5.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f}},
			{{5.0f, 0.0f, 5.0f}, {0.0f, 1.0f, 0.0f}, {1.0f, 0.0f}},
			{{5.0f, 0.0f, -5.0f}, {0.0f, 1.0f, 0.0f}, {1.0f, 1.0f}},
			{{-5.0f, 0.0f, -5.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 1.0f}},
		};

		std::vector<uint32_t> indices = {0, 1, 2, 2, 3, 0};

		plane_mesh.vertex_buffer = new veekay::graphics::Buffer(
			vertices.size() * sizeof(Vertex), vertices.data(), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);

		plane_mesh.index_buffer = new veekay::graphics::Buffer(
			indices.size() * sizeof(uint32_t), indices.data(), VK_BUFFER_USAGE_INDEX_BUFFER_BIT);

		plane_mesh.indices = uint32_t(indices.size());
	}

	{
		std::vector<Vertex> vertices = {
			{{-0.5f, -0.5f, -0.5f}, {0.0f, 0.0f, -1.0f}, {0.0f, 0.0f}},
			{{+0.5f, -0.5f, -0.5f}, {0.0f, 0.0f, -1.0f}, {1.0f, 0.0f}},
			{{+0.5f, +0.5f, -0.5f}, {0.0f, 0.0f, -1.0f}, {1.0f, 1.0f}},
			{{-0.5f, +0.5f, -0.5f}, {0.0f, 0.0f, -1.0f}, {0.0f, 1.0f}},

			{{+0.5f, -0.5f, -0.5f}, {1.0f, 0.0f, 0.0f}, {0.0f, 0.0f}},
			{{+0.5f, -0.5f, +0.5f}, {1.0f, 0.0f, 0.0f}, {1.0f, 0.0f}},
			{{+0.5f, +0.5f, +0.5f}, {1.0f, 0.0f, 0.0f}, {1.0f, 1.0f}},
			{{+0.5f, +0.5f, -0.5f}, {1.0f, 0.0f, 0.0f}, {0.0f, 1.0f}},

			{{+0.5f, -0.5f, +0.5f}, {0.0f, 0.0f, 1.0f}, {0.0f, 0.0f}},
			{{-0.5f, -0.5f, +0.5f}, {0.0f, 0.0f, 1.0f}, {1.0f, 0.0f}},
			{{-0.5f, +0.5f, +0.5f}, {0.0f, 0.0f, 1.0f}, {1.0f, 1.0f}},
			{{+0.5f, +0.5f, +0.5f}, {0.0f, 0.0f, 1.0f}, {0.0f, 1.0f}},

			{{-0.5f, -0.5f, +0.5f}, {-1.0f, 0.0f, 0.0f}, {0.0f, 0.0f}},
			{{-0.5f, -0.5f, -0.5f}, {-1.0f, 0.0f, 0.0f}, {1.0f, 0.0f}},
			{{-0.5f, +0.5f, -0.5f}, {-1.0f, 0.0f, 0.0f}, {1.0f, 1.0f}},
			{{-0.5f, +0.5f, +0.5f}, {-1.0f, 0.0f, 0.0f}, {0.0f, 1.0f}},

			{{-0.5f, -0.5f, +0.5f}, {0.0f, -1.0f, 0.0f}, {0.0f, 0.0f}},
			{{+0.5f, -0.5f, +0.5f}, {0.0f, -1.0f, 0.0f}, {1.0f, 0.0f}},
			{{+0.5f, -0.5f, -0.5f}, {0.0f, -1.0f, 0.0f}, {1.0f, 1.0f}},
			{{-0.5f, -0.5f, -0.5f}, {0.0f, -1.0f, 0.0f}, {0.0f, 1.0f}},

			{{-0.5f, +0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f}},
			{{+0.5f, +0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}, {1.0f, 0.0f}},
			{{+0.5f, +0.5f, +0.5f}, {0.0f, 1.0f, 0.0f}, {1.0f, 1.0f}},
			{{-0.5f, +0.5f, +0.5f}, {0.0f, 1.0f, 0.0f}, {0.0f, 1.0f}},
		};

		std::vector<uint32_t> indices = {
			0, 1, 2, 2, 3, 0,
			4, 5, 6, 6, 7, 4,
			8, 9, 10, 10, 11, 8,
			12, 13, 14, 14, 15, 12,
			16, 17, 18, 18, 19, 16,
			20, 21, 22, 22, 23, 20,
		};

		cube_mesh.vertex_buffer = new veekay::graphics::Buffer(
			vertices.size() * sizeof(Vertex), vertices.data(), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);

		cube_mesh.index_buffer = new veekay::graphics::Buffer(
			indices.size() * sizeof(uint32_t), indices.data(), VK_BUFFER_USAGE_INDEX_BUFFER_BIT);

		cube_mesh.indices = uint32_t(indices.size());
	}

	models.emplace_back(Model{
		.mesh = plane_mesh,
		.transform = {},
		.material = { {0.8f,0.8f,0.8f}, 0, {0.5f,0.5f,0.5f}, 16.0f }
	});

	models.emplace_back(Model{
		.mesh = cube_mesh,
		.transform = { .position = {-2.0f, -0.5f, -1.5f} },
		.material = { {1.0f,0.0f,0.0f}, 0, {1.0f,1.0f,1.0f}, 64.0f }
	});

	models.emplace_back(Model{
		.mesh = cube_mesh,
		.transform = { .position = {1.5f, -0.5f, -0.5f} },
		.material = { {0.0f,1.0f,0.0f}, 0, {1.0f,1.0f,1.0f}, 64.0f }
	});

	models.emplace_back(Model{
		.mesh = cube_mesh,
		.transform = { .position = {0.0f, -0.5f, 1.0f} },
		.material = { {0.0f,0.0f,1.0f}, 0, {1.0f,1.0f,1.0f}, 64.0f }
	});
}

void shutdown() {
	VkDevice& device = veekay::app.vk_device;

	vkDestroySampler(device, missing_texture_sampler, nullptr);
	delete missing_texture;

	delete cube_mesh.index_buffer;
	delete cube_mesh.vertex_buffer;

	delete plane_mesh.index_buffer;
	delete plane_mesh.vertex_buffer;

	delete light_ssbo_buffer;
	delete model_uniforms_buffer;
	delete scene_uniforms_buffer;

	vkDestroyDescriptorSetLayout(device, descriptor_set_layout, nullptr);
	vkDestroyDescriptorPool(device, descriptor_pool, nullptr);

	vkDestroyPipeline(device, pipeline, nullptr);
	vkDestroyPipelineLayout(device, pipeline_layout, nullptr);
	vkDestroyShaderModule(device, fragment_shader_module, nullptr);
	vkDestroyShaderModule(device, vertex_shader_module, nullptr);
}

void update(double time) {
    // --- LIGHTING UI (Остается без изменений) ---
    ImGui::Begin("Lighting");

    ImGui::Text("Ambient");
    ImGui::ColorEdit3("Color##Amb", &ambient_light.color.x);

    ImGui::Separator();
    ImGui::Text("Directional");
    ImGui::SliderFloat3("Dir", &directional_light.direction.x, -1.0f, 1.0f); 
    directional_light.direction = veekay::vec3::normalized(directional_light.direction);
    ImGui::ColorEdit3("Color##Dir", &directional_light.color.x);

    ImGui::Separator();
    ImGui::Text("Point Lights (2)");
    light_ssbo.point_light_count = 2;

    // Point Light 0
    ImGui::PushID(0);
    ImGui::SliderFloat3("Pos##0", &light_ssbo.point_lights[0].position.x, -5.0f, 5.0f); 
    ImGui::ColorEdit3("Color##0", &light_ssbo.point_lights[0].color.x);
    ImGui::SliderFloat("Const##0", &light_ssbo.point_lights[0].constant, 0.0f, 2.0f);
    ImGui::SliderFloat("Lin##0", &light_ssbo.point_lights[0].linear, 0.0f, 1.0f);
    ImGui::SliderFloat("Quad##0", &light_ssbo.point_lights[0].quadratic, 0.0f, 1.0f);
    ImGui::PopID();

    // Point Light 1
    ImGui::PushID(1);
    ImGui::SliderFloat3("Pos##1", &light_ssbo.point_lights[1].position.x, -5.0f, 5.0f); 
    ImGui::ColorEdit3("Color##1", &light_ssbo.point_lights[1].color.x);
    ImGui::SliderFloat("Const##1", &light_ssbo.point_lights[1].constant, 0.0f, 2.0f);
    ImGui::SliderFloat("Lin##1", &light_ssbo.point_lights[1].linear, 0.0f, 1.0f);
    ImGui::SliderFloat("Quad##1", &light_ssbo.point_lights[1].quadratic, 0.0f, 1.0f);
    ImGui::PopID();

    // Spot Light UI
    ImGui::Separator();
    ImGui::Text("Spot Light");
    ImGui::SliderFloat3("Pos##Spot", &spot_light.position.x, -5.0f, 5.0f); 
    ImGui::SliderFloat3("Dir##Spot", &spot_light.direction.x, -1.0f, 1.0f); 
    spot_light.direction = veekay::vec3::normalized(spot_light.direction);
    ImGui::ColorEdit3("Color##Spot", &spot_light.color.x);
    ImGui::SliderFloat("Inner CutOff (Deg)", &spot_light.inner_cutOff, 0.0f, 45.0f);
    ImGui::SliderFloat("Outer CutOff (Deg)", &spot_light.outer_cutOff, 0.0f, 45.0f);

    ImGui::End();

    // КОПИРУЕМ SSBO В GPU
    *(LightSSBO*)light_ssbo_buffer->mapped_region = light_ssbo;

    // --- CAMERA UI ---
    ImGui::Begin("Camera"); 

    bool old_mode = camera.is_look_at;
    ImGui::Checkbox("Use Look-At Mode", &camera.is_look_at);

    // --- УПРОЩЕННАЯ ЛОГИКА СОХРАНЕНИЯ СОСТОЯНИЯ (СБРОС ВРАЩЕНИЯ) ---
    // Убираем сложную тригонометрию, которая вызывала "хуйню".
    if (old_mode != camera.is_look_at) {
        // Если перешли обратно в Transformation Mode, сбрасываем вращение
        // Это предотвратит скачок, используя известное состояние (0, 0, 0).
        if (!camera.is_look_at) {
             camera.rotation = {0.0f, 0.0f, 0.0f}; 
        }
    }

    // Если Look-At активен, показать цель
    if (camera.is_look_at) {
        ImGui::Text("Look-At Target");
        ImGui::SliderFloat3("Target Pos", &camera.target.x, -5.0f, 5.0f); 
    } else {
         // В режиме Трансформации можно показать текущие углы
        ImGui::Text("Rotation (Euler)");
        ImGui::Text("Pitch: %.2f, Yaw: %.2f", camera.rotation.x, camera.rotation.y);
    }
    
    ImGui::End();

    // БЛОКИРУЕМ КАМЕРУ, КОГДА ImGui АКТИВНО
    if (ImGui::IsWindowHovered(ImGuiHoveredFlags_AnyWindow)) {
        return;
    }

    // --- КАМЕРА (ЛОГИКА УПРАВЛЕНИЯ, ОСТАЕТСЯ ПРЕЖНЕЙ) ---
    
    // Вращение (только в режиме Трансформации)
    if (!camera.is_look_at && veekay::input::mouse::isButtonDown(veekay::input::mouse::Button::left)) {
        auto delta = veekay::input::mouse::cursorDelta();
        camera.rotation.y -= delta.x * 0.15f;
        camera.rotation.x -= delta.y * 0.15f;
        camera.rotation.x = std::clamp(camera.rotation.x, -89.0f, 89.0f);
    }

    // Движение (работает в обоих режимах)
    veekay::mat4 view = camera.view();
    // Извлекаем Right и Front векторы из View Matrix
    veekay::vec3 right = veekay::vec3::normalized({ view[0][0], view[1][0], view[2][0] });
    // Front инвертирован, т.к. view matrix создана для Right-Handed System (Vulkan)
    veekay::vec3 front = veekay::vec3::normalized({ -view[0][2], -view[1][2], -view[2][2] }); 

    float speed = 0.05f;
    if (veekay::input::keyboard::isKeyDown(veekay::input::keyboard::Key::w)) camera.position += front * speed;
    if (veekay::input::keyboard::isKeyDown(veekay::input::keyboard::Key::s)) camera.position -= front * speed;
    if (veekay::input::keyboard::isKeyDown(veekay::input::keyboard::Key::d)) camera.position += right * speed;
    if (veekay::input::keyboard::isKeyDown(veekay::input::keyboard::Key::a)) camera.position -= right * speed;
    if (veekay::input::keyboard::isKeyDown(veekay::input::keyboard::Key::q)) camera.position.y += speed;
    if (veekay::input::keyboard::isKeyDown(veekay::input::keyboard::Key::z)) camera.position.y -= speed;

    // --- UNIFORMS ---
    float aspect = float(veekay::app.window_width) / float(veekay::app.window_height);
    *(SceneUniforms*)scene_uniforms_buffer->mapped_region = { camera.view_projection(aspect) };

    std::vector<ModelUniforms> mu(models.size());
    for (size_t i = 0; i < models.size(); ++i) {
        mu[i].model = models[i].transform.matrix();
        mu[i].albedo_color = models[i].material.albedo;
        mu[i].specular_color = models[i].material.specular;
        mu[i].shininess = models[i].material.shininess;
    }
    std::copy(mu.begin(), mu.end(), (ModelUniforms*)model_uniforms_buffer->mapped_region);
}

void render(VkCommandBuffer cmd, VkFramebuffer framebuffer) {
	vkResetCommandBuffer(cmd, 0);

	VkCommandBufferBeginInfo begin_info{
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
		.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
	};
	vkBeginCommandBuffer(cmd, &begin_info);

	VkClearValue clear_color{.color = {{0.1f, 0.1f, 0.1f, 1.0f}}};
	VkClearValue clear_depth{.depthStencil = {1.0f, 0}};
	VkClearValue clear_values[] = {clear_color, clear_depth};

	VkRenderPassBeginInfo rp_info{
		.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
		.renderPass = veekay::app.vk_render_pass,
		.framebuffer = framebuffer,
		.renderArea = { .extent = {veekay::app.window_width, veekay::app.window_height} },
		.clearValueCount = 2,
		.pClearValues = clear_values,
	};
	vkCmdBeginRenderPass(cmd, &rp_info, VK_SUBPASS_CONTENTS_INLINE);

	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
	VkDeviceSize zero_offset = 0;

	VkBuffer current_vertex_buffer = VK_NULL_HANDLE;
	VkBuffer current_index_buffer = VK_NULL_HANDLE;

	for (size_t i = 0; i < models.size(); ++i) {
		const Model& model = models[i];
		const Mesh& mesh = model.mesh;

		if (current_vertex_buffer != mesh.vertex_buffer->buffer) {
			current_vertex_buffer = mesh.vertex_buffer->buffer;
			vkCmdBindVertexBuffers(cmd, 0, 1, &current_vertex_buffer, &zero_offset);
		}

		if (current_index_buffer != mesh.index_buffer->buffer) {
			current_index_buffer = mesh.index_buffer->buffer;
			vkCmdBindIndexBuffer(cmd, current_index_buffer, zero_offset, VK_INDEX_TYPE_UINT32);
		}

		uint32_t dyn_offset = i * sizeof(ModelUniforms);
		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout, 0, 1, &descriptor_set, 1, &dyn_offset);

		struct Push { 
            veekay::vec3 cam; float _p0; 
            veekay::vec3 amb; float _p1; 
            veekay::vec3 dir; float _p2; 
            veekay::vec3 dcol; float _p3; 
            
            // Spot Light Data (НОВЫЕ ПОЛЯ)
            veekay::vec3 s_pos; float _s_p0;
            veekay::vec3 s_dir; float _s_p1;
            veekay::vec3 s_col; float _s_p2;
            float s_inner;
            float s_outer;
            float _s_p3;
            float _s_p4;
        };
		Push push = {
            camera.position, 0,
            ambient_light.color, 0,
            directional_light.direction, 0,
            directional_light.color, 0,
            // НОВЫЕ ДАННЫЕ
            spot_light.position, 0,
            spot_light.direction, 0,
            spot_light.color, 0,
            (float)cos(toRadians(spot_light.inner_cutOff)), // передаем косинусы
            (float)cos(toRadians(spot_light.outer_cutOff)),
            0, 0
        };
        vkCmdPushConstants(cmd, pipeline_layout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(push), &push);

		vkCmdDrawIndexed(cmd, mesh.indices, 1, 0, 0, 0);
	}

	vkCmdEndRenderPass(cmd);
	vkEndCommandBuffer(cmd);
}

} // namespace

int main() {
	return veekay::run({
		.init = initialize,
		.shutdown = shutdown,
		.update = update,
		.render = render,
	});
}