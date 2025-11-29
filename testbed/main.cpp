#include "veekay/input.hpp"
#include <cstdint>
#include <climits>
#include <vector>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <cstring> 
#include <GLFW/glfw3.h>

#define _USE_MATH_DEFINES
#include <math.h>
#include <cmath>
#include <veekay/veekay.hpp>

#include <imgui.h>
#include <vulkan/vulkan_core.h>
#include <lodepng.h>

veekay::graphics::Texture* load_texture_from_file(const char* path, VkCommandBuffer cmd) {
    unsigned width, height;
    std::vector<unsigned char> image_data;
    unsigned error = lodepng::decode(image_data, width, height, path);
    if (error) {
        std::cerr << "LodePNG error: " << lodepng_error_text(error) << std::endl;
        return nullptr;
    }
    return new veekay::graphics::Texture(cmd, width, height, VK_FORMAT_R8G8B8A8_UNORM, image_data.data());
}


constexpr float PI = 3.14159265359f;
inline float to_radians(float degrees) { return degrees * (PI / 180.0f); }
inline float to_degrees(float radians) { return radians * (180.0f / PI); }

veekay::mat4 ortho(float left, float right, float bottom, float top, float near, float far) {
    veekay::mat4 result = {};
    result[0][0] = 2.0f / (right - left);
    result[1][1] = 2.0f / (top - bottom);
    result[2][2] = -1.0f / (far - near);
    result[3][0] = -(right + left) / (right - left);
    result[3][1] = -(top + bottom) / (top - bottom);
    result[3][2] = -near / (far - near);
    result[3][3] = 1.0f;
    return result;
}


namespace {

constexpr uint32_t max_models = 1024;
constexpr uint32_t max_point_lights = 8;

constexpr uint32_t SHADOW_MAP_SIZE = 2048;

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


    veekay::graphics::Texture* albedo_texture;
    veekay::graphics::Texture* specular_texture;
    veekay::graphics::Texture* emissive_texture;
    VkSampler sampler;
    VkDescriptorSet descriptor_set;
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
	veekay::vec3 direction = {0.0f, -1.0f, 0.0f};
	veekay::vec3 color = {1.0f, 1.0f, 1.0f};
	float inner_cutOff = 12.5f;
	float outer_cutOff = 17.5f;

	float constant = 1.0f;
    float linear = 0.14f;
    float quadratic = 0.07f;
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

	VkPipelineLayout pipeline_layout;

	VkPipeline pipeline;

	veekay::graphics::Texture* shadow_map_texture;
	VkSampler shadow_map_sampler;
	VkShaderModule shadow_vertex_shader_module;
	VkPipelineLayout shadow_pipeline_layout;
	VkPipeline shadow_pipeline;
	VkRenderPass shadow_render_pass;
	VkFramebuffer shadow_framebuffer;

	veekay::graphics::Buffer* scene_uniforms_buffer;
	veekay::graphics::Buffer* model_uniforms_buffer;
	veekay::graphics::Buffer* light_ssbo_buffer;

	Mesh plane_mesh;
	Mesh cube_mesh;

    veekay::graphics::Texture *texture_lenna, *texture_checker;
    veekay::graphics::Texture *texture_white, *texture_black, *texture_emissive_example;
    VkSampler sampler_linear, sampler_nearest;
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


	
	shadow_map_texture = new veekay::graphics::Texture(cmd, SHADOW_MAP_SIZE, SHADOW_MAP_SIZE, VK_FORMAT_D32_SFLOAT, nullptr, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);

	VkSamplerCreateInfo shadow_sampler_info {
		.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO, .magFilter = VK_FILTER_LINEAR, .minFilter = VK_FILTER_LINEAR,
		.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER, .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER, .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER,
		.compareEnable = VK_TRUE, .compareOp = VK_COMPARE_OP_LESS_OR_EQUAL, .borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE
	};
	vkCreateSampler(device, &shadow_sampler_info, nullptr, &shadow_map_sampler);



    texture_lenna = load_texture_from_file("assets/lenna.png", cmd);

    uint32_t checker_pixels[] = { 0xffffffff, 0xff000000, 0xff000000, 0xffffffff };
	
    texture_checker = new veekay::graphics::Texture(cmd, 2, 2, VK_FORMAT_R8G8B8A8_UNORM, checker_pixels);
    uint32_t white_pixel = 0xffffffff;
    texture_white = new veekay::graphics::Texture(cmd, 1, 1, VK_FORMAT_R8G8B8A8_UNORM, &white_pixel);
    uint32_t black_pixel = 0xff000000;
    texture_black = new veekay::graphics::Texture(cmd, 1, 1, VK_FORMAT_R8G8B8A8_UNORM, &black_pixel);
    uint32_t emissive_pixel = 0xff00ffff;
    texture_emissive_example = new veekay::graphics::Texture(cmd, 1, 1, VK_FORMAT_R8G8B8A8_UNORM, &emissive_pixel);

    VkSamplerCreateInfo sampler_info_linear{ .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO, .magFilter = VK_FILTER_LINEAR, .minFilter = VK_FILTER_LINEAR, .addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT, .addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT, .addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT };
    vkCreateSampler(device, &sampler_info_linear, nullptr, &sampler_linear);
    VkSamplerCreateInfo sampler_info_nearest{ .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO, .magFilter = VK_FILTER_NEAREST, .minFilter = VK_FILTER_NEAREST, .addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT, .addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT, .addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT };
    vkCreateSampler(device, &sampler_info_nearest, nullptr, &sampler_nearest);

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

	shadow_vertex_shader_module = loadShaderModule("./shaders/shadow.vert.spv");
	if (!shadow_vertex_shader_module) {
		std::cerr << "Failed to load Vulkan shadow vertex shader from file\n";
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
			{ VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, max_models * 4 } // теперь 4 
		};

		VkDescriptorPoolCreateInfo info{
			.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
			.maxSets = max_models,
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
			{ 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT },
            { 3, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT },
            { 4, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT },
            { 5, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT },
			{ 6, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT } // теперь так
		};

		VkDescriptorSetLayoutCreateInfo info{
			.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
			.bindingCount = 7,
			.pBindings = bindings,
		};

		if (vkCreateDescriptorSetLayout(device, &info, nullptr, &descriptor_set_layout) != VK_SUCCESS) {
			std::cerr << "Failed to create Vulkan descriptor set layout\n";
			veekay::app.running = false;
			return;
		}
	}

	

// vvvv ВСТАВЬ ЭТОТ БЛОК НА МЕСТО УДАЛЕННЫХ vvvv
	// --- Создание Render Pass и Framebuffer для теней ---
    {
        VkAttachmentDescription depthAttachment{};
        depthAttachment.format = shadow_map_texture->format;
        depthAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
        depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        depthAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        depthAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        depthAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        depthAttachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;

        VkAttachmentReference depthAttachmentRef{};
        depthAttachmentRef.attachment = 0;
        depthAttachmentRef.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

        VkSubpassDescription subpass{};
        subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass.pDepthStencilAttachment = &depthAttachmentRef;

        VkRenderPassCreateInfo renderPassInfo{};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        renderPassInfo.attachmentCount = 1;
        renderPassInfo.pAttachments = &depthAttachment;
        renderPassInfo.subpassCount = 1;
        renderPassInfo.pSubpasses = &subpass;

        if (vkCreateRenderPass(device, &renderPassInfo, nullptr, &shadow_render_pass) != VK_SUCCESS) {
            throw std::runtime_error("failed to create shadow render pass!");
        }

        VkFramebufferCreateInfo framebufferInfo{};
        framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        framebufferInfo.renderPass = shadow_render_pass;
        framebufferInfo.attachmentCount = 1;
        framebufferInfo.pAttachments = &shadow_map_texture->view;
        framebufferInfo.width = SHADOW_MAP_SIZE;
        framebufferInfo.height = SHADOW_MAP_SIZE;
        framebufferInfo.layers = 1;

        if (vkCreateFramebuffer(device, &framebufferInfo, nullptr, &shadow_framebuffer) != VK_SUCCESS) {
            throw std::runtime_error("failed to create shadow framebuffer!");
        }
    }

	// --- Создание РАЗДЕЛЬНЫХ Pipeline Layouts ---
    // Layout для ОСНОВНОГО конвейера (с push-константой для обоих шейдеров)
    {
        VkPushConstantRange push_range{ .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, .offset = 0, .size = 256 };
        VkPipelineLayoutCreateInfo layout_info{ .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, .setLayoutCount = 1, .pSetLayouts = &descriptor_set_layout, .pushConstantRangeCount = 1, .pPushConstantRanges = &push_range, };
		if (vkCreatePipelineLayout(device, &layout_info, nullptr, &pipeline_layout) != VK_SUCCESS) { std::cerr << "Failed to create pipeline layout\n"; return; }
    }
    // Layout для ТЕНЕВОГО конвейера (БЕЗ push-констант)
    {
        VkPushConstantRange shadow_push_range{ 
    .stageFlags = VK_SHADER_STAGE_VERTEX_BIT, 
    .offset = 96, 
    .size = 64 
};
VkPipelineLayoutCreateInfo shadow_layout_info{ 
    .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, 
    .setLayoutCount = 1, 
    .pSetLayouts = &descriptor_set_layout,
};
		if (vkCreatePipelineLayout(device, &shadow_layout_info, nullptr, &shadow_pipeline_layout) != VK_SUCCESS) { std::cerr << "Failed to create shadow pipeline layout\n"; return; }
    }


    // --- Создание конвейера для теней (shadow_pipeline) ---
    {
        VkPipelineShaderStageCreateInfo shadow_stage_info { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_VERTEX_BIT, .module = shadow_vertex_shader_module, .pName = "main" };
		VkVertexInputBindingDescription shadow_buffer_binding{.binding=0, .stride=sizeof(Vertex), .inputRate=VK_VERTEX_INPUT_RATE_VERTEX};
		VkVertexInputAttributeDescription shadow_attributes[] = {{.location=0, .binding=0, .format=VK_FORMAT_R32G32B32_SFLOAT, .offset=offsetof(Vertex, position)}};
		VkPipelineVertexInputStateCreateInfo shadow_input_state_info{.sType=VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO, .vertexBindingDescriptionCount=1, .pVertexBindingDescriptions=&shadow_buffer_binding, .vertexAttributeDescriptionCount=1, .pVertexAttributeDescriptions=shadow_attributes};
        VkPipelineViewportStateCreateInfo viewportState { .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO, .viewportCount = 1, .scissorCount = 1 };
        VkDynamicState dynamicStates[] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
        VkPipelineDynamicStateCreateInfo dynamicState { .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO, .dynamicStateCount = 2, .pDynamicStates = dynamicStates };
        VkPipelineColorBlendStateCreateInfo shadow_blend_info { .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO }; // Пустой, т.к. цвета нет
        
        VkGraphicsPipelineCreateInfo pipeline_info {
            .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, .stageCount = 1, .pStages = &shadow_stage_info,
            .pVertexInputState = &shadow_input_state_info, .pInputAssemblyState = &assembly_state_info, .pViewportState = &viewportState,
            .pRasterizationState = &raster_info, .pMultisampleState = &sample_info, .pDepthStencilState = &depth_info, .pColorBlendState = &shadow_blend_info,
            .pDynamicState = &dynamicState, .layout = shadow_pipeline_layout, .renderPass = shadow_render_pass
        };
        if (vkCreateGraphicsPipelines(device, nullptr, 1, &pipeline_info, nullptr, &shadow_pipeline) != VK_SUCCESS) {
            std::cerr << "Failed to create shadow pipeline\n";
            veekay::app.running = false;
            return;
        }
    }
// ^^^^

	VkGraphicsPipelineCreateInfo info{
		.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
		.stageCount = 2,
		.pStages = stage_infos, // будут использоваться 2 шейдерные программы (vertex_shader_module и fragment_shader_module)
		.pVertexInputState = &input_state_info, // связываемся с шейдеров
		.pInputAssemblyState = &assembly_state_info,
		.pViewportState = &viewport_info,
		.pRasterizationState = &raster_info, // настройки растеризации
		.pMultisampleState = &sample_info,
		.pDepthStencilState = &depth_info, // настраиваем глубину
		.pColorBlendState = &blend_info,
		.layout = pipeline_layout,
		.renderPass = veekay::app.vk_render_pass,
	};

	if (vkCreateGraphicsPipelines(device, nullptr, 1, &info, nullptr, &pipeline) != VK_SUCCESS) {
		std::cerr << "Failed to create Vulkan pipeline\n";
		veekay::app.running = false;
		return;
	}

	// создание экземпляра класса Buffer (создаем на GPU быстрый буфер 64 байта для хранения view_projection)
	scene_uniforms_buffer = new veekay::graphics::Buffer(
		sizeof(SceneUniforms), nullptr, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);

	// просим у VRAM кусок памяти для хранения информации о 1024 моделях (Dynamic Uniform Buffer)
	model_uniforms_buffer = new veekay::graphics::Buffer(
		max_models * veekay::graphics::Buffer::structureAlignment(sizeof(ModelUniforms)),
		nullptr, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT); // будем использовать этот буфер как Uniform Buffer

	// создание буферы для хранения массива точечных источников ( Storage Buffer)
	light_ssbo_buffer = new veekay::graphics::Buffer(
		sizeof(LightSSBO), nullptr, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT); // подсказка драйверу, что что буфер может быть большим и к нему будут обращаться их шейдера

	// инициализируем 2 источника света
    light_ssbo.point_lights[0] = { {2.0f, 1.0f, 0.0f}, 0.0f, {1.0f, 0.0f, 0.0f}, 0.0f, 1.0f, 0.14f, 0.07f };
    light_ssbo.point_lights[1] = { {-2.0f, 1.0f, 0.0f}, 0.0f, {0.0f, 1.0f, 0.0f}, 0.0f, 1.0f, 0.14f, 0.07f };
    light_ssbo.point_light_count = 2;

	// синхронизация с GPU (копируем из локальной переменной в память на GPU)
    *(LightSSBO*)light_ssbo_buffer->mapped_region = light_ssbo;

	{
		// 3 элемент везд - структурные координаты
		std::vector<Vertex> vertices = {
			{{-5.0f, 0.0f, 5.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 5.0f}}, 
			{{5.0f, 0.0f, 5.0f}, {0.0f, 1.0f, 0.0f}, {5.0f, 5.0f}},
			{{5.0f, 0.0f, -5.0f}, {0.0f, 1.0f, 0.0f}, {5.0f, 0.0f}},
			{{-5.0f, 0.0f, -5.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f}},
		};
		std::vector<uint32_t> indices = {0, 1, 2, 2, 3, 0};
		plane_mesh.vertex_buffer = new veekay::graphics::Buffer(vertices.size() * sizeof(Vertex), vertices.data(), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
		plane_mesh.index_buffer = new veekay::graphics::Buffer(indices.size() * sizeof(uint32_t), indices.data(), VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
		plane_mesh.indices = uint32_t(indices.size());
	}

	{
		// создание геометрии куба
		std::vector<Vertex> vertices = {
			{{-0.5f, -0.5f, -0.5f}, {0.0f, 0.0f, -1.0f}, {0.0f, 0.0f}},{{+0.5f, -0.5f, -0.5f}, {0.0f, 0.0f, -1.0f}, {1.0f, 0.0f}},{{+0.5f, +0.5f, -0.5f}, {0.0f, 0.0f, -1.0f}, {1.0f, 1.0f}},{{-0.5f, +0.5f, -0.5f}, {0.0f, 0.0f, -1.0f}, {0.0f, 1.0f}},
			{{+0.5f, -0.5f, -0.5f}, {1.0f, 0.0f, 0.0f}, {0.0f, 0.0f}},{{+0.5f, -0.5f, +0.5f}, {1.0f, 0.0f, 0.0f}, {1.0f, 0.0f}},{{+0.5f, +0.5f, +0.5f}, {1.0f, 0.0f, 0.0f}, {1.0f, 1.0f}},{{+0.5f, +0.5f, -0.5f}, {1.0f, 0.0f, 0.0f}, {0.0f, 1.0f}},
			{{+0.5f, -0.5f, +0.5f}, {0.0f, 0.0f, 1.0f}, {0.0f, 0.0f}},{{-0.5f, -0.5f, +0.5f}, {0.0f, 0.0f, 1.0f}, {1.0f, 0.0f}},{{-0.5f, +0.5f, +0.5f}, {0.0f, 0.0f, 1.0f}, {1.0f, 1.0f}},{{+0.5f, +0.5f, +0.5f}, {0.0f, 0.0f, 1.0f}, {0.0f, 1.0f}},
			{{-0.5f, -0.5f, +0.5f}, {-1.0f, 0.0f, 0.0f}, {0.0f, 0.0f}},{{-0.5f, -0.5f, -0.5f}, {-1.0f, 0.0f, 0.0f}, {1.0f, 0.0f}},{{-0.5f, +0.5f, -0.5f}, {-1.0f, 0.0f, 0.0f}, {1.0f, 1.0f}},{{-0.5f, +0.5f, +0.5f}, {-1.0f, 0.0f, 0.0f}, {0.0f, 1.0f}},
			{{-0.5f, -0.5f, +0.5f}, {0.0f, -1.0f, 0.0f}, {0.0f, 0.0f}},{{+0.5f, -0.5f, +0.5f}, {0.0f, -1.0f, 0.0f}, {1.0f, 0.0f}},{{+0.5f, -0.5f, -0.5f}, {0.0f, -1.0f, 0.0f}, {1.0f, 1.0f}},{{-0.5f, -0.5f, -0.5f}, {0.0f, -1.0f, 0.0f}, {0.0f, 1.0f}},
			{{-0.5f, +0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f}},{{+0.5f, +0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}, {1.0f, 0.0f}},{{+0.5f, +0.5f, +0.5f}, {0.0f, 1.0f, 0.0f}, {1.0f, 1.0f}},{{-0.5f, +0.5f, +0.5f}, {0.0f, 1.0f, 0.0f}, {0.0f, 1.0f}},
		};
		std::vector<uint32_t> indices = { 0, 1, 2, 2, 3, 0, 4, 5, 6, 6, 7, 4, 8, 9, 10, 10, 11, 8, 12, 13, 14, 14, 15, 12, 16, 17, 18, 18, 19, 16, 20, 21, 22, 22, 23, 20, };
		cube_mesh.vertex_buffer = new veekay::graphics::Buffer( vertices.size() * sizeof(Vertex), vertices.data(), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
		cube_mesh.index_buffer = new veekay::graphics::Buffer( indices.size() * sizeof(uint32_t), indices.data(), VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
		cube_mesh.indices = uint32_t(indices.size());
	}

	models.emplace_back(Model{
		.mesh = plane_mesh, .transform = {}, .material = {{1.0f,1.0f,1.0f}, 0, {0.1f,0.1f,0.1f}, 4.0f },
        .albedo_texture = texture_checker, .specular_texture = texture_white, .emissive_texture = texture_black, .sampler = sampler_nearest
	});

	// левый куб использует texture_lenna и sampler_linear
	// основная текстура - texture_lenna (lenna.png)
	// .sampler = sampler_linear - сэмплер, сглаженные пиксели (билинейная фильтрация)
	models.emplace_back(Model{
		.mesh = cube_mesh, .transform = { .position = {-2.0f, -0.5f, -1.5f} }, .material = {{1.0f,1.0f,1.0f}, 0, {1.0f,1.0f,1.0f}, 64.0f },
        .albedo_texture = texture_lenna, .specular_texture = texture_white, .emissive_texture = texture_black, .sampler = sampler_linear
	});

	// правый куб (матовый)
	// Карта бликов: черная (МАТОВАЯ поверхность)
	// Карта свечения: черная (не светится)
	// Сэмплер: четкие пиксели
	models.emplace_back(Model{
		.mesh = cube_mesh, .transform = { .position = {1.5f, -0.5f, -0.5f} }, .material = {{1.0f,1.0f,1.0f}, 0, {1.0f,1.0f,1.0f}, 128.0f },
        .albedo_texture = texture_checker, .specular_texture = texture_black, .emissive_texture = texture_black, .sampler = sampler_nearest
	});

	// центральный куб (светящийся)
	// .albedo_texture = texture_white, основная текстура: белая (просто чтобы был базовый цвет)
	// .specular_texture = texture_white, карта бликов: белая (глянцевый)
	// .emissive_texture = texture_emissive_example, карта свечения: яркая (бирюзовая/желтая)
	models.emplace_back(Model{
		.mesh = cube_mesh, .transform = { .position = {0.0f, -0.5f, 1.0f} }, .material = {{1.0f,1.0f,1.0f}, 0, {1.0f,1.0f,1.0f}, 64.0f },
        .albedo_texture = texture_white, .specular_texture = texture_white, .emissive_texture = texture_emissive_example, .sampler = sampler_linear
	});

    // настраиваем персональные Description Set в цикле
	// создаем VkWriteDescriptorSet, который связывает конкретную текстуру 
	// (model.albedo_texture) и сэмплер (model.sampler) с привязкой 3 в наборе дескрипторов этой модели.
	// проходим по всем моделям и для каждой заполняем ее descriptor_set
    for (auto& model : models) {
        VkDescriptorSetAllocateInfo alloc_info {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, .descriptorPool = descriptor_pool,
            .descriptorSetCount = 1, .pSetLayouts = &descriptor_set_layout,
        };
		// выделяем под дескриптор сет память
        if (vkAllocateDescriptorSets(device, &alloc_info, &model.descriptor_set) != VK_SUCCESS) {
            throw std::runtime_error("failed to allocate descriptor sets!");
        }
        VkDescriptorBufferInfo scene_buffer_info{ scene_uniforms_buffer->buffer, 0, sizeof(SceneUniforms) };
        VkDescriptorBufferInfo model_buffer_info{ model_uniforms_buffer->buffer, 0, sizeof(ModelUniforms) };
        VkDescriptorBufferInfo light_ssbo_info{ light_ssbo_buffer->buffer, 0, VK_WHOLE_SIZE };

		// указываем, какую текстуру и какой сэмплер взять
        VkDescriptorImageInfo albedo_image_info{ model.sampler, model.albedo_texture->view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
        VkDescriptorImageInfo specular_image_info{ model.sampler, model.specular_texture->view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
        VkDescriptorImageInfo emissive_image_info{ model.sampler, model.emissive_texture->view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
		
		
		VkDescriptorImageInfo shadow_image_info { shadow_map_sampler, shadow_map_texture->view, VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL };
		
		// создаем запись для привязки (связываем реальную текстуру и сэмплер с этим адресом)
        VkWriteDescriptorSet write_sets[] = {
            { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, model.descriptor_set, 0, 0, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, nullptr, &scene_buffer_info },
            { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, model.descriptor_set, 1, 0, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, nullptr, &model_buffer_info },
            { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, model.descriptor_set, 2, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &light_ssbo_info },
            { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, model.descriptor_set, 3, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &albedo_image_info, nullptr },
            { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, model.descriptor_set, 4, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &specular_image_info, nullptr },
            { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, model.descriptor_set, 5, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &emissive_image_info, nullptr },
			{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, model.descriptor_set, 6, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &shadow_image_info, nullptr }
        };

		// заполняем дескриптор сет данными нашей модели
        vkUpdateDescriptorSets(device, 7, write_sets, 0, nullptr);
    }
}

void shutdown() {
    VkDevice& device = veekay::app.vk_device;

    // Очистка ресурсов для теней
    vkDestroyFramebuffer(device, shadow_framebuffer, nullptr);
    vkDestroyRenderPass(device, shadow_render_pass, nullptr);
    delete shadow_map_texture;
    vkDestroySampler(device, shadow_map_sampler, nullptr);
    vkDestroyPipeline(device, shadow_pipeline, nullptr);
    // shadow_pipeline_layout не удаляем, так как он равен pipeline_layout
    vkDestroyShaderModule(device, shadow_vertex_shader_module, nullptr);
	vkDestroyPipelineLayout(device, shadow_pipeline_layout, nullptr);

    
    // Старая очистка
    delete texture_lenna; delete texture_checker; delete texture_white; delete texture_black; delete texture_emissive_example;
    vkDestroySampler(device, sampler_linear, nullptr); vkDestroySampler(device, sampler_nearest, nullptr);
    delete cube_mesh.index_buffer; delete cube_mesh.vertex_buffer;
    delete plane_mesh.index_buffer; delete plane_mesh.vertex_buffer;
    delete light_ssbo_buffer; delete model_uniforms_buffer; delete scene_uniforms_buffer;
    vkDestroyDescriptorSetLayout(device, descriptor_set_layout, nullptr);
    vkDestroyDescriptorPool(device, descriptor_pool, nullptr);
    vkDestroyPipeline(device, pipeline, nullptr);
    vkDestroyPipelineLayout(device, pipeline_layout, nullptr);
    vkDestroyShaderModule(device, fragment_shader_module, nullptr);
    vkDestroyShaderModule(device, vertex_shader_module, nullptr);
}

// работа CPU
void update(double time) {
    
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

	ImGui::Text("Spot Light Attenuation");
	ImGui::SliderFloat("Const##Spot", &spot_light.constant, 0.0f, 2.0f); 
	ImGui::SliderFloat("Lin##Spot", &spot_light.linear, 0.0f, 1.0f);     
	ImGui::SliderFloat("Quad##Spot", &spot_light.quadratic, 0.0f, 1.0f); 

    ImGui::End();

    // КОПИРУЕМ SSBO В GPU (синхронизация с GPU)
	// mapped_region - указатель типа void* на область памяти
	// Эта память "видна" и CPU, и GPU (с помощью флага VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)
	// просто копируем байты из структуры в этот участок памяти
    *(LightSSBO*)light_ssbo_buffer->mapped_region = light_ssbo;

    ImGui::Begin("Camera");

    bool old_mode = camera.is_look_at;
    ImGui::Checkbox("Use Look-At Mode", &camera.is_look_at);

    if (old_mode != camera.is_look_at) {
        if (!camera.is_look_at) {
             camera.rotation = {0.0f, 0.0f, 0.0f};
        }
    }

    if (camera.is_look_at) {
        ImGui::Text("Look-At Target");
        ImGui::SliderFloat3("Target Pos", &camera.target.x, -5.0f, 5.0f);
    } else {
        ImGui::Text("Rotation (Euler)");
        ImGui::Text("Pitch: %.2f, Yaw: %.2f", camera.rotation.x, camera.rotation.y);
    }

    ImGui::End();

    if (ImGui::IsWindowHovered(ImGuiHoveredFlags_AnyWindow)) {
        return;
    }

    if (!camera.is_look_at && veekay::input::mouse::isButtonDown(veekay::input::mouse::Button::left)) {
        auto delta = veekay::input::mouse::cursorDelta();
        camera.rotation.y -= delta.x * 0.15f;
        camera.rotation.x -= delta.y * 0.15f;
        camera.rotation.x = std::clamp(camera.rotation.x, -89.0f, 89.0f);
    }

    veekay::mat4 view = camera.view();
    veekay::vec3 right = veekay::vec3::normalized({ view[0][0], view[1][0], view[2][0] });
    veekay::vec3 front = veekay::vec3::normalized({ -view[0][2], -view[1][2], -view[2][2] });

    float speed = 0.05f;

    if (veekay::input::keyboard::isKeyDown(veekay::input::keyboard::Key::w)) camera.position += front * speed;
    if (veekay::input::keyboard::isKeyDown(veekay::input::keyboard::Key::s)) camera.position -= front * speed;
    if (veekay::input::keyboard::isKeyDown(veekay::input::keyboard::Key::d)) camera.position += right * speed;
    if (veekay::input::keyboard::isKeyDown(veekay::input::keyboard::Key::a)) camera.position -= right * speed;
    if (veekay::input::keyboard::isKeyDown(veekay::input::keyboard::Key::q)) camera.position.y += speed;
    if (veekay::input::keyboard::isKeyDown(veekay::input::keyboard::Key::z)) camera.position.y -= speed;

    float aspect = float(veekay::app.window_width) / float(veekay::app.window_height);
    *(SceneUniforms*)scene_uniforms_buffer->mapped_region = { camera.view_projection(aspect) };

    
	const size_t alignment =
		veekay::graphics::Buffer::structureAlignment(sizeof(ModelUniforms));

	for (size_t i = 0, n = models.size(); i < n; ++i) {
		ModelUniforms uniforms;
        uniforms.model = models[i].transform.matrix();
        uniforms.albedo_color = models[i].material.albedo; // копируем цвет
        uniforms.specular_color = models[i].material.specular;
        uniforms.shininess = models[i].material.shininess;

		//вычисляем точный адрес внутри большого буфера, куда нужно положить данные
		char* const pointer = static_cast<char*>(model_uniforms_buffer->mapped_region) + i * alignment;
		memcpy(pointer, &uniforms, sizeof(ModelUniforms)); // копируем по вычисленному адресу (в общей памяти, доступной GPU, лежит блок данных для куба)
	}
}

//записываем последовательность команд в cmd
void render(VkCommandBuffer cmd, VkFramebuffer framebuffer) {
	vkResetCommandBuffer(cmd, 0);
	VkCommandBufferBeginInfo begin_info{.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
	vkBeginCommandBuffer(cmd, &begin_info);

    veekay::mat4 light_space_matrix;
    // --- ПРОХОД 1: РЕНДЕР КАРТЫ ТЕНЕЙ (SHADOW PASS) ---
    {
        float near_plane = 1.0f, far_plane = 15.0f;
        veekay::mat4 light_projection = ortho(-10.0f, 10.0f, -10.0f, 10.0f, near_plane, far_plane);
        veekay::mat4 light_view = veekay::mat4::look_at(-directional_light.direction * 5.0f, {0.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f});
        light_space_matrix = light_view * light_projection;

        *(SceneUniforms*)scene_uniforms_buffer->mapped_region = { light_space_matrix };

        VkClearValue clear_depth{.depthStencil = {1.0f, 0}};
        VkRenderPassBeginInfo rp_info{};
        rp_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        rp_info.renderPass = shadow_render_pass;
        rp_info.framebuffer = shadow_framebuffer;
        rp_info.renderArea.offset = {0, 0};
        rp_info.renderArea.extent = {SHADOW_MAP_SIZE, SHADOW_MAP_SIZE}; // ИСПРАВЛЕНИЕ: Указываем размер
        rp_info.clearValueCount = 1;
        rp_info.pClearValues = &clear_depth;
        vkCmdBeginRenderPass(cmd, &rp_info, VK_SUBPASS_CONTENTS_INLINE);

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, shadow_pipeline);
        VkViewport viewport { .width = SHADOW_MAP_SIZE, .height = SHADOW_MAP_SIZE, .minDepth = 0.0f, .maxDepth = 1.0f };
        VkRect2D scissor { .extent = {SHADOW_MAP_SIZE, SHADOW_MAP_SIZE} };
        vkCmdSetViewport(cmd, 0, 1, &viewport);
        vkCmdSetScissor(cmd, 0, 1, &scissor);

        VkDeviceSize zero_offset = 0;
        const size_t model_uniforms_alignment = veekay::graphics::Buffer::structureAlignment(sizeof(ModelUniforms));
        for (size_t i = 0; i < models.size(); ++i) {
            const Model& model = models[i];
            vkCmdBindVertexBuffers(cmd, 0, 1, &model.mesh.vertex_buffer->buffer, &zero_offset);
            vkCmdBindIndexBuffer(cmd, model.mesh.index_buffer->buffer, zero_offset, VK_INDEX_TYPE_UINT32);
            uint32_t dyn_offset = i * model_uniforms_alignment;
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, shadow_pipeline_layout, 0, 1, &model.descriptor_set, 1, &dyn_offset);
            
            

            vkCmdDrawIndexed(cmd, model.mesh.indices, 1, 0, 0, 0);
        }
        vkCmdEndRenderPass(cmd);
    }

    // --- ПРОХОД 2: ОСНОВНОЙ РЕНДЕР СЦЕНЫ (MAIN PASS) ---
    {
        float aspect = float(veekay::app.window_width) / float(veekay::app.window_height);
		*(SceneUniforms*)scene_uniforms_buffer->mapped_region = { camera.view_projection(aspect) };

        VkRect2D renderArea = {};
        renderArea.offset = {0, 0};
        renderArea.extent = {veekay::app.window_width, veekay::app.window_height};
        VkClearValue clear_color{.color = {{0.1f, 0.1f, 0.1f, 1.0f}}};
        VkClearValue clear_depth{.depthStencil = {1.0f, 0}};
        VkClearValue clear_values[] = {clear_color, clear_depth};
        VkRenderPassBeginInfo rp_info{.sType=VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO, .renderPass=veekay::app.vk_render_pass, .framebuffer=framebuffer, .renderArea=renderArea, .clearValueCount=2, .pClearValues=clear_values};
        vkCmdBeginRenderPass(cmd, &rp_info, VK_SUBPASS_CONTENTS_INLINE);

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
		VkDeviceSize zero_offset = 0;
		VkBuffer current_vertex_buffer = VK_NULL_HANDLE;
		VkBuffer current_index_buffer = VK_NULL_HANDLE;
		const size_t model_uniforms_alignment =
			veekay::graphics::Buffer::structureAlignment(sizeof(ModelUniforms));

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
            uint32_t dyn_offset = i * model_uniforms_alignment;
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout, 0, 1, &model.descriptor_set, 1, &dyn_offset);

            // ИСПРАВЛЕНИЕ: полная структура Push, включая матрицу света
            struct Push {
                veekay::vec3 cam; float time;
                veekay::vec3 amb; float _p1;
                veekay::vec3 dir; float _p2;
                veekay::vec3 dcol; float _p3;
                veekay::mat4 light_space_matrix; // Добавлена матрица
                veekay::vec3 s_pos; float _s_p0;
                veekay::vec3 s_dir; float _s_p1;
                veekay::vec3 s_col; float _s_p2;
                float s_inner; float s_outer;
                float s_const; float s_lin; float s_quad;
                float _s_p3; float _s_p4;
            };
            Push push = {
                camera.position, (float)glfwGetTime(),
                ambient_light.color, 0,
                directional_light.direction, 0,
                directional_light.color, 0,
                light_space_matrix, // Передаем матрицу
                spot_light.position, 0,
                spot_light.direction, 0,
                spot_light.color, 0,
                (float)cos(toRadians(spot_light.inner_cutOff)),
                (float)cos(toRadians(spot_light.outer_cutOff)),
                spot_light.constant,
                spot_light.linear,
                spot_light.quadratic,
                0, 0
            };
            vkCmdPushConstants(cmd, pipeline_layout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(push), &push);
            vkCmdDrawIndexed(cmd, mesh.indices, 1, 0, 0, 0);
        }
        vkCmdEndRenderPass(cmd);
    }
    vkEndCommandBuffer(cmd);
}

}

int main() {
	return veekay::run({
		.init = initialize,
		.shutdown = shutdown,
		.update = update,
		.render = render,
	});
}