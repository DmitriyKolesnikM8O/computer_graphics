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

// загружаем изображение из файла
veekay::graphics::Texture* load_texture_from_file(const char* path, VkCommandBuffer cmd) {
    unsigned width, height;
    std::vector<unsigned char> image_data;
    unsigned error = lodepng::decode(image_data, width, height, path);
    if (error) {
        std::cerr << "LodePNG error: " << lodepng_error_text(error) << std::endl;
        return nullptr;
    }
	// создаем текстуру
    return new veekay::graphics::Texture(cmd, width, height, VK_FORMAT_R8G8B8A8_UNORM, image_data.data());
}


constexpr float PI = 3.14159265359f;
inline float to_radians(float degrees) { return degrees * (PI / 180.0f); }
inline float to_degrees(float radians) { return radians * (180.0f / PI); }

namespace {

constexpr uint32_t max_models = 1024;
constexpr uint32_t max_point_lights = 8;

struct Vertex {
	veekay::vec3 position;
	veekay::vec3 normal;
	veekay::vec2 uv; // вершины объектов должны содержать текстурные координаты
};

// глобальные данные для всей сцены
struct SceneUniforms {
	veekay::mat4 view_projection; // комбинированная матрица вида и проекции (где и под каким углом мы видим сцену)
};

// Внешний вид поверхности объекта
struct Material {
	veekay::vec3 albedo = {1.0f, 1.0f, 1.0f};
	float _pad0;
	veekay::vec3 specular = {1.0f, 1.0f, 1.0f};
	float shininess = 32.0f;
};

// Как данные об одном объекте лежат в GPU
struct ModelUniforms {
	veekay::mat4 model;
	veekay::vec3 albedo_color;
	float shininess;
	veekay::vec3 specular_color;
	float _pad;
};

// описываем геометрию объекта	
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

// описываем один полноценный объект

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

	veekay::mat4 view() const; // вычисляем и возвращаем матрицу вида
	veekay::mat4 view_projection(float aspect_ratio) const; // матрица вида * матрицу проекции


    veekay::vec3 target = {0.0f, -0.5f, 0.0f}; // точка в пространстве, куда смотрит камера (только если включен is_look_at)
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

	float constant = 1.0f;
    float linear = 0.14f;
    float quadratic = 0.07f;
};

struct LightSSBO {
	PointLight point_lights[max_point_lights]; // данные для каждого точечного источника света
	uint32_t point_light_count = 0; // сколько из 8 источников реально используем
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
	// обертка вокруг SPIR-V кода. Просто ресурс, который подключаю к конвейеру
	VkShaderModule vertex_shader_module;
	VkShaderModule fragment_shader_module;

	// фабрика для создания реальных наборов дескрипторов
	VkDescriptorPool descriptor_pool;

	// описание всей структуры данных, которую ожидает шейдер: UBO, Dynamic UBO и SSBO
	VkDescriptorSetLayout descriptor_set_layout;

	// это нам больше не нужно
	// VkDescriptorSet descriptor_set;

	// описание интерфейса между шейдерами и данными, передаваемыми из кода
	VkPipelineLayout pipeline_layout;

	// сконфигурированный графический конвейер
	VkPipeline pipeline;

	// хранит данные, общие для всей сцены (матрица view_projection)
	veekay::graphics::Buffer* scene_uniforms_buffer;
	// хранит данные, уникальные для каждого объекта (model и свойства)
	veekay::graphics::Buffer* model_uniforms_buffer;
	// массив данных о точечных источниках света
	veekay::graphics::Buffer* light_ssbo_buffer;

	// группируем вместе 2 указателя на буферы (vertex_buffer и index_buffer)
	Mesh plane_mesh;
	Mesh cube_mesh;


	// veekay::graphics::Texture* missing_texture;
	// VkSampler missing_texture_sampler;
    veekay::graphics::Texture *texture_lenna, *texture_checker;
    veekay::graphics::Texture *texture_white, *texture_black, *texture_emissive_example;
    VkSampler sampler_linear, sampler_nearest;
}

float toRadians(float degrees) {
	return degrees * float(M_PI) / 180.0f;
}

// вычисляем и возвращаем матрицу модели
veekay::mat4 Transform::matrix() const {
	auto t = veekay::mat4::translation(position);
	auto s = veekay::mat4::scaling(scale);
	auto rx = veekay::mat4::rotation({1,0,0}, toRadians(rotation.x));
	auto ry = veekay::mat4::rotation({0,1,0}, toRadians(rotation.y));
	auto rz = veekay::mat4::rotation({0,0,1}, toRadians(rotation.z));
	return t * rz * ry * rx * s;
}

veekay::mat4 Camera::view() const {
	// доп задание в лабе 2
    if (is_look_at) {
        return veekay::mat4::look_at(position, target, {0.0f, 1.0f, 0.0f});
    }

	// вычисляем матрицу вида через обратные переносы и повороты
	// camera.rotation меняется в update() на основе движения мыши
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
		.codeSize = size, // размер байт-кода шейдера
		.pCode = buffer.data(), // указатель на байт код шейдера
	};

	VkShaderModule result;
	// veekay::app.vk_device - логическое устройство GPU
	// &info - указатель на VkShaderModuleCreateInfo (где инфа о шейдере)
	// nullptr - указатель на аллокатор памяти (используем базовый)
	if (vkCreateShaderModule(veekay::app.vk_device, &info, nullptr, &result) != VK_SUCCESS) {
		return nullptr;
	}
	return result;
}

void initialize(VkCommandBuffer cmd) {
	VkDevice& device = veekay::app.vk_device;

	// загружаем текстуру и создаем сэмплеры
    texture_lenna = load_texture_from_file("assets/lenna.png", cmd);
    uint32_t checker_pixels[] = { 0xffffffff, 0xff000000, 0xff000000, 0xffffffff };
    texture_checker = new veekay::graphics::Texture(cmd, 2, 2, VK_FORMAT_R8G8B8A8_UNORM, checker_pixels);
    uint32_t white_pixel = 0xffffffff;
    texture_white = new veekay::graphics::Texture(cmd, 1, 1, VK_FORMAT_R8G8B8A8_UNORM, &white_pixel);
    uint32_t black_pixel = 0xff000000;
    texture_black = new veekay::graphics::Texture(cmd, 1, 1, VK_FORMAT_R8G8B8A8_UNORM, &black_pixel);
    uint32_t emissive_pixel = 0xff00ffff;
    texture_emissive_example = new veekay::graphics::Texture(cmd, 1, 1, VK_FORMAT_R8G8B8A8_UNORM, &emissive_pixel);

	// создаем 2 сэмплера (linear - для сглаженного (размытого) изображения; nearest - для четкого, пиксельного отображения)
	// addressModeV{U} - заставляет текстуру повторяться
    VkSamplerCreateInfo sampler_info_linear{ .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO, .magFilter = VK_FILTER_LINEAR, .minFilter = VK_FILTER_LINEAR, .addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT, .addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT, .addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT };
    vkCreateSampler(device, &sampler_info_linear, nullptr, &sampler_linear);
    VkSamplerCreateInfo sampler_info_nearest{ .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO, .magFilter = VK_FILTER_NEAREST, .minFilter = VK_FILTER_NEAREST, .addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT, .addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT, .addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT };
    vkCreateSampler(device, &sampler_info_nearest, nullptr, &sampler_nearest);

	// загружаем не GLSL-код, а SPIR-V (бинарный, платформонезависимый промежуточный код)
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

	// данные для вершин
	VkVertexInputBindingDescription buffer_binding{
		.binding = 0, // идут одним потоком
		.stride = sizeof(Vertex), // размер одного пакета данных
		.inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
	};

	// внутри пакета данных
	VkVertexInputAttributeDescription attributes[] = {
		// 3 VK_FORMAT_R32G32B32_SFLOAT со смещением offsetof(Vertex, position)
		{ .location = 0, .binding = 0, .format = VK_FORMAT_R32G32B32_SFLOAT, .offset = offsetof(Vertex, position) },
		// 3 float'а со смещением offsetof(Vertex, normal)
		{ .location = 1, .binding = 0, .format = VK_FORMAT_R32G32B32_SFLOAT, .offset = offsetof(Vertex, normal) },
		// 2 float'а со смещением offsetof(Vertex, uv)
		{ .location = 2, .binding = 0, .format = VK_FORMAT_R32G32_SFLOAT, .offset = offsetof(Vertex, uv) },
	};

	// описываем формат вершинных данных
	VkPipelineVertexInputStateCreateInfo input_state_info{
		.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
		.vertexBindingDescriptionCount = 1, // 1 буфер для данных вершин
		.pVertexBindingDescriptions = &buffer_binding,
		.vertexAttributeDescriptionCount = 3, // у каждой вершины 3 атрибута
		.pVertexAttributeDescriptions = attributes,
	};

	// как собирать вершины в примитивы
	VkPipelineInputAssemblyStateCreateInfo assembly_state_info{
		.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
		.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, // берем вершины из буфера по 3 штуки и каждую тройку считаем независимым треугольником
	};

	// настраиваем растеризацию
	VkPipelineRasterizationStateCreateInfo raster_info{
		.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
		.polygonMode = VK_POLYGON_MODE_FILL, //сплошным цветом заливаем
		.cullMode = VK_CULL_MODE_BACK_BIT, //не рисуй треугольники, которые повернуты задней стороной
		.frontFace = VK_FRONT_FACE_CLOCKWISE,
		.lineWidth = 1.0f,
	};

	// настраиваем мультисэмплинг
	VkPipelineMultisampleStateCreateInfo sample_info{
		.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
		.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT, //выключен, юзаем 1 сэмпл на пиксель
	};

	// преобразование из нормализованных координат устройства в экранные координаты
	VkViewport viewport{
		.width = static_cast<float>(veekay::app.window_width),
		.height = static_cast<float>(veekay::app.window_height),
		.minDepth = 0.0f,
		.maxDepth = 1.0f,
	};

	// область отсечения
	VkRect2D scissor{
		.extent = {veekay::app.window_width, veekay::app.window_height},
	};

	// говорим растеризатору, как отображать финальные координаты на реальные пиксели окна
	VkPipelineViewportStateCreateInfo viewport_info{
		.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
		.viewportCount = 1,
		.pViewports = &viewport, // правило для трансформации координат
		.scissorCount = 1,
		.pScissors = &scissor,
	};

	VkPipelineDepthStencilStateCreateInfo depth_info{
		.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
		.depthTestEnable = true, 
		.depthWriteEnable = true,
		.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL, // пиксель будет нарисован, только если его глубина меньше или равна глубине уже нарисованного пикселя в этой точке
	};

	// как 1 цветовой буфер должен обрабатывать результат из фрагментного шейдера
	VkPipelineColorBlendAttachmentState attachment_info{
		// VK_COLOR_COMPONENT_R_BIT - разрешить в красный и ....
		.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
	};

	// настраиваем смешение
	VkPipelineColorBlendStateCreateInfo blend_info{
		.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
		.logicOpEnable = false, //отключен, то есть просто перезаписывается цвет
		.attachmentCount = 1,
		.pAttachments = &attachment_info
	};

	{
		/*
			резервируем место под:
				- 8 дескрипторов UNIFORM_BUFFER ...
		*/
		VkDescriptorPoolSize pools[] = {
			{ VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 8 },
			{ VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 8 },
			{ VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 8 },
			{ VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, max_models * 3 } 
		};

		// описываем конфигурацию пула дескрипторов
		VkDescriptorPoolCreateInfo info{
			.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
			.maxSets = max_models,
			.poolSizeCount = 4,
			.pPoolSizes = pools,
		};

		// выделяем внутри драйвера память под хранение запрошенных дескрипторов
		if (vkCreateDescriptorPool(device, &info, nullptr, &descriptor_pool) != VK_SUCCESS) {
			std::cerr << "Failed to create Vulkan descriptor pool\n";
			veekay::app.running = false;
			return;
		}
	}

	{
		// массив, где каждый элемент описывает 1 ресурс, доступный шейдеру
		// В шейдерах будет ресурс в binding = 0. Это будет UNIFORM_BUFFER. Он будет один (descriptorCount = 1). Доступ к нему нужен и в вершинном, и во фрагментном шейдере
		// + привязки для текстур (в слоте 3 COMBINED_IMAGE_SAMPLER)
		VkDescriptorSetLayoutBinding bindings[] = {
			{ 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT },
			{ 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT }, // для Description Set будет SSBO
			{ 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT },
            { 3, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT },
            { 4, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT },
            { 5, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT }
		};

		// описываем, как будет выглядеть 1 Descriptor Set
		VkDescriptorSetLayoutCreateInfo info{
			.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
			.bindingCount = 6,
			.pBindings = bindings,
		};

		// создаем VkDescriptorSetLayout
		if (vkCreateDescriptorSetLayout(device, &info, nullptr, &descriptor_set_layout) != VK_SUCCESS) {
			std::cerr << "Failed to create Vulkan descriptor set layout\n";
			veekay::app.running = false;
			return;
		}
	}

	// это больше не нужно
	// {
	// 	VkDescriptorSetAllocateInfo info{...};
	// 	if (vkAllocateDescriptorSets(device, &info, &descriptor_set) != VK_SUCCESS) {...}
	// }

	VkPushConstantRange push_range{
		.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
		.size = 144
	};

	// способы, которыми мы будем передавать данные в шейдеры (кроме вершинных атрибутов)
	VkPipelineLayoutCreateInfo layout_info{
		.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
		.setLayoutCount = 1, // 1 Description Set
		.pSetLayouts = &descriptor_set_layout,
		.pushConstantRangeCount = 1, // 1 блок push констант
		.pPushConstantRanges = &push_range,
	};

	// создаем VkPipelineLayout (описывает все данные, которые шейдеры могут запросить)
	if (vkCreatePipelineLayout(device, &layout_info, nullptr, &pipeline_layout) != VK_SUCCESS) {
		std::cerr << "Failed to create Vulkan pipeline layout\n";
		veekay::app.running = false;
		return;
	}

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

	// создание экземпляра класса  экземпляра класса Buffer (создаем на GPU быстрый буфер 64 байта для хранения view_projection)
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

	// Инициализирую модели с новыми полями
	
	// пол использует texture_checker и sampler_nearest
	// plane_mesh - геометрия плоскости
	// Основная текстура texture_checker - черно-белая шахматка
	// specular_texture = texture_white - карта бликов: белая (супер глянцевая поверхность)
	// .emissive_texture = texture_black - карта свечения: черная (не светится)
	// .sampler = sampler_nearest - сэмплер: четкие пиксели, без размытия
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
    for (auto& model : models) {
        VkDescriptorSetAllocateInfo alloc_info {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, .descriptorPool = descriptor_pool,
            .descriptorSetCount = 1, .pSetLayouts = &descriptor_set_layout,
        };
        if (vkAllocateDescriptorSets(device, &alloc_info, &model.descriptor_set) != VK_SUCCESS) {
            throw std::runtime_error("failed to allocate descriptor sets!");
        }
        VkDescriptorBufferInfo scene_buffer_info{ scene_uniforms_buffer->buffer, 0, sizeof(SceneUniforms) };
        VkDescriptorBufferInfo model_buffer_info{ model_uniforms_buffer->buffer, 0, sizeof(ModelUniforms) };
        VkDescriptorBufferInfo light_ssbo_info{ light_ssbo_buffer->buffer, 0, VK_WHOLE_SIZE };
        VkDescriptorImageInfo albedo_image_info{ model.sampler, model.albedo_texture->view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
        VkDescriptorImageInfo specular_image_info{ model.sampler, model.specular_texture->view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
        VkDescriptorImageInfo emissive_image_info{ model.sampler, model.emissive_texture->view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
        VkWriteDescriptorSet write_sets[] = {
            { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, model.descriptor_set, 0, 0, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, nullptr, &scene_buffer_info },
            { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, model.descriptor_set, 1, 0, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, nullptr, &model_buffer_info },
            { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, model.descriptor_set, 2, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &light_ssbo_info },
            { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, model.descriptor_set, 3, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &albedo_image_info, nullptr },
            { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, model.descriptor_set, 4, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &specular_image_info, nullptr },
            { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, model.descriptor_set, 5, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &emissive_image_info, nullptr }
        };
        vkUpdateDescriptorSets(device, 6, write_sets, 0, nullptr);
    }
}

void shutdown() {
	VkDevice& device = veekay::app.vk_device;

	
	// vkDestroySampler(device, missing_texture_sampler, nullptr);
	// delete missing_texture;
    delete texture_lenna; 
    delete texture_checker;
    delete texture_white; 
    delete texture_black; 
    delete texture_emissive_example;
    vkDestroySampler(device, sampler_linear, nullptr);
    vkDestroySampler(device, sampler_nearest, nullptr);

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
	
	// перемещение
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
	vkResetCommandBuffer(cmd, 0); // очистка

	VkCommandBufferBeginInfo begin_info{
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
		.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, // используем буфер только 1 раз, потом перезапишем
	};
	vkBeginCommandBuffer(cmd, &begin_info); // начало записи

	VkClearValue clear_color{.color = {{0.1f, 0.1f, 0.1f, 1.0f}}};
	VkClearValue clear_depth{.depthStencil = {1.0f, 0}};
	VkClearValue clear_values[] = {clear_color, clear_depth};

	// все параметры для начала рендеринга
	VkRenderPassBeginInfo rp_info{
		.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
		.renderPass = veekay::app.vk_render_pass,
		.framebuffer = framebuffer,
		.renderArea = { .extent = {veekay::app.window_width, veekay::app.window_height} }, // все окно
		.clearValueCount = 2, // 2 значения для очистки (цвет и глубина)
		.pClearValues = clear_values,
	};

	// начать процесс рендеринга
	vkCmdBeginRenderPass(cmd, &rp_info, VK_SUBPASS_CONTENTS_INLINE);

	// теперь знаем, какие шейдеры использовать, как рисовать треугольники и так далее
	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
	VkDeviceSize zero_offset = 0;

	VkBuffer current_vertex_buffer = VK_NULL_HANDLE;
	VkBuffer current_index_buffer = VK_NULL_HANDLE;

	// MERGE: Используем более правильный способ получения смещения из нового каркаса
	const size_t model_uniforms_alignment =
		veekay::graphics::Buffer::structureAlignment(sizeof(ModelUniforms));

	for (size_t i = 0; i < models.size(); ++i) {
		const Model& model = models[i];
		const Mesh& mesh = model.mesh;

		if (current_vertex_buffer != mesh.vertex_buffer->buffer) {
			current_vertex_buffer = mesh.vertex_buffer->buffer;

			//Для binding = 0 во входных данных вершин использовать вот этот vertex_buffer
			vkCmdBindVertexBuffers(cmd, 0, 1, &current_vertex_buffer, &zero_offset);
		}

		if (current_index_buffer != mesh.index_buffer->buffer) {
			current_index_buffer = mesh.index_buffer->buffer;

			//Вершины нужно соединять в треугольники согласно индексам из вот этого index_buffer
			vkCmdBindIndexBuffer(cmd, current_index_buffer, zero_offset, VK_INDEX_TYPE_UINT32);
		}

		uint32_t dyn_offset = i * model_uniforms_alignment; // MERGE: Используем выровненное смещение (вычисляем то же самое смещение, что в update)

		
		// vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout, 0, 1, &descriptor_set, 1, &dyn_offset);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout, 0, 1, &model.descriptor_set, 1, &dyn_offset);

		// Добавляем время в пуш-константы
		struct Push {
            veekay::vec3 cam; float time;
            veekay::vec3 amb; float _p1;
            veekay::vec3 dir; float _p2;
            veekay::vec3 dcol; float _p3;

            veekay::vec3 s_pos; float _s_p0;
            veekay::vec3 s_dir; float _s_p1;
            veekay::vec3 s_col; float _s_p2;
            float s_inner;
            float s_outer;
			float s_const; float s_lin; float s_quad; 
            float _s_p3;
            float _s_p4;
        };
		Push push = {
			// glfwGetTime - возвращает количество секунд, прошедших с момента запуска программы.
            camera.position, (float)glfwGetTime(),  //передача времени в шейдер (там используется для анимации, чтобы пол плыл)
            ambient_light.color, 0,
            directional_light.direction, 0,
            directional_light.color, 0,

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

		// самый быстрый способ передачи данных, минуя буферы, просто ложим в регистры GPU
        vkCmdPushConstants(cmd, pipeline_layout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(push), &push);

		// нарисуй объект, состоящий из mesh.indices вершин (по итогу: отдали GPU приказ нарисовать куб и указали, где в памяти лежат данные о его материале)
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