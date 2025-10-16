#include <cstdint>
#include <climits>
#include <vector>
#include <iostream>
#include <fstream>
#include <cmath>
#include <cstring> // Для memcpy

#include <veekay/veekay.hpp>

#include <imgui.h>
#include <vulkan/vulkan_core.h>


/*
Цилиндр с ортографической проекцией. Сгенерировать цилиндр (~50
вершин, без крышек). Цвет задается через uniform/push constant.
Использовать ортографическую проекцию. Анимация: цилиндр
перемещается по траектории (например, по окружности). Добавить UI
элементы для изменения радиуса траектории.

Допки:

1. Сложная траектория анимации. Заменить простую анимацию
(например, вращение или синусоидальное движение) на более сложную
траекторию. Например, объект движется по спирали, эллипсу или
траектории в форме восьмерки (используя параметрические уравнения).
Реализовать через модельную матрицу с зависимостью от времени.

2. Переключение между перспективной и ортографической проекцией.
Добавить возможность переключения между перспективной и
ортографической проекцией с помощью элементов UI. Реализовать
через изменение проекционной матрицы.

3. Анимация с паузой и реверсом. Добавить управление анимацией: UI
элемент для паузы/возобновления и UI элемент для реверса
направления (например, изменение знака угловой скорости вращения).
Реализовать через переменную времени и модификатор направления в
модельной матрице.
*/


// Для M_PI
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace {

// --- СТРУКТУРЫ ---
struct Matrix {
	float m[4][4];
};

struct Vector {
	float x, y, z;
};

struct Vertex {
	Vector position;
	// NOTE: You can add more attributes
};

struct ShaderConstants {
	Matrix projection;
	Matrix transform;
	Vector color;
};

struct VulkanBuffer {
	VkBuffer buffer;
	VkDeviceMemory memory;
};
// -----------------------------------------------------------------------------

// --- ГЛОБАЛЬНЫЕ КОНСТАНТЫ И ПЕРЕМЕННЫЕ ЛАБОРАТОРНОЙ ---
constexpr float camera_fov = 70.0f;
constexpr float camera_near_plane = 0.01f;
constexpr float camera_far_plane = 100.0f; 

// --- ВАЖНО: сделаем симметричный Z-диапазон для ортографической камеры,
// чтобы объекты с отрицательным Z (например, z = -5.0f) были внутри объёма ---
constexpr float ORTHO_NEAR = -10.0f; 
constexpr float ORTHO_FAR = 10.0f; 

enum class ProjectionType { PERSPECTIVE, ORTHOGRAPHIC };
ProjectionType current_projection = ProjectionType::ORTHOGRAPHIC; // По умолчанию ортографическая
constexpr int CYLINDER_SEGMENTS = 100;

// Глобальные переменные для трансформаций и анимации
Vector model_position = {0.0f, 0.0f, -5.0f}; // Z по умолчанию -5 для видимости
float model_rotation = 0.0f; // Ручное вращение
Vector model_color = {0.5f, 1.0f, 0.7f };
bool model_spin = true;

// НОВЫЕ/ИЗМЕНЕННЫЕ ПЕРЕМЕННЫЕ ДЛЯ АНИМАЦИИ
float trajectory_scale = 2.0f; // Масштаб для траектории "восьмерки"
float ortho_scale = 5.0f;

// --- Главное исправление скорости анимации ---
// Теперь animation_speed управляет *масштабом времени* (time scale).
// По умолчанию 0.2 — адекватно медленно; UI позволяет от 0 до 2.0.
float animation_speed = 0.2f;

float current_time = 0.0f; // Текущее "время" для параметризации траектории
bool animation_paused = false; // Состояние паузы
float animation_direction = 1.0f; // Направление: 1.0f (вперед) или -1.0f (назад/реверс)

// Уменьшенный базовый множитель скорости вращения
float spin_speed_multiplier = 0.05f; 

float cylinder_tilt = 0.3f; // Наклон цилиндра для лучшего обзора

// Vulkan Buffers and Modules
VkShaderModule vertex_shader_module;
VkShaderModule fragment_shader_module;
VkPipelineLayout pipeline_layout;
VkPipeline pipeline;

VulkanBuffer vertex_buffer;
VulkanBuffer index_buffer;
uint32_t index_count = 0; // Хранить количество индексов цилиндра

// --- ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ---

Matrix identity() {
	Matrix result{};
	// инициализация нулями - уже выполнена при value-initialization
	result.m[0][0] = 1.0f;
	result.m[1][1] = 1.0f;
	result.m[2][2] = 1.0f;
	result.m[3][3] = 1.0f;
	return result;
}

// Ортографическая проекция: использует scale и aspect_ratio для задания границ X/Y
// Мы ожидаем: right = scale * aspect_ratio, top = scale, left = -right, bottom = -top
// Z преобразуется из [near_z, far_z] -> [0, 1] (Vulkan)
Matrix orthographic(float scale, float aspect_ratio, float near_z, float far_z) {
	Matrix result{};

	float right = scale * aspect_ratio;
	float top = scale;

	// X,Y: симметрично
	result.m[0][0] = 1.0f / right; // 2/(r-l) / 2 => since symmetric, 1/right maps [-right,right] -> [-1,1]
	result.m[1][1] = 1.0f / top;

	// Z: map [near, far] -> [0,1] для Vulkan
	// formula: z_ndc = (z - near) / (far - near)
	// we place it into matrix as m[2][2] and m[3][2]
	result.m[2][2] = 1.0f / (far_z - near_z);
	result.m[3][2] = -near_z / (far_z - near_z);

	// Последний элемент
	result.m[3][3] = 1.0f;

	return result;
}

// Выбирает проекцию: ортографическая или перспективная
Matrix projection(float fov, float aspect_ratio, float near, float far) {
	Matrix result{};

    if (current_projection == ProjectionType::ORTHOGRAPHIC) {
		return orthographic(ortho_scale, aspect_ratio, ORTHO_NEAR, ORTHO_FAR); 
	}
    
	const float radians = fov * (float)M_PI / 180.0f;
	const float cot = 1.0f / tanf(radians / 2.0f);

	result.m[0][0] = cot / aspect_ratio;
	result.m[1][1] = cot;
    
    // Z-компонент для Vulkan: map z from [near,far] to [0,1]
	result.m[2][2] = far / (near - far);
	result.m[3][2] = -(far * near) / (far - near);
	
	result.m[2][3] = -1.0f;

	return result;
}

Matrix translation(Vector vector) {
	Matrix result = identity();

	result.m[3][0] = vector.x;
	result.m[3][1] = vector.y;
	result.m[3][2] = vector.z;

	return result;
}

Matrix rotation(Vector axis, float angle) {
	Matrix result{};

	float length = sqrtf(axis.x * axis.x + axis.y * axis.y + axis.z * axis.z);
	if (length < 1e-6) return identity();

	axis.x /= length;
	axis.y /= length;
	axis.z /= length;

	float sina = sinf(angle);
	float cosa = cosf(angle);
	float cosv = 1.0f - cosa;

	result.m[0][0] = (axis.x * axis.x * cosv) + cosa;
	result.m[0][1] = (axis.x * axis.y * cosv) + (axis.z * sina);
	result.m[0][2] = (axis.x * axis.z * cosv) - (axis.y * sina);

	result.m[1][0] = (axis.y * axis.x * cosv) - (axis.z * sina);
	result.m[1][1] = (axis.y * axis.y * cosv) + cosa;
	result.m[1][2] = (axis.y * axis.z * cosv) + (axis.x * sina);

	result.m[2][0] = (axis.z * axis.x * cosv) + (axis.y * sina);
	result.m[2][1] = (axis.z * axis.y * cosv) - (axis.x * sina);
	result.m[2][2] = (axis.z * axis.z * cosv) + cosa;

	result.m[3][3] = 1.0f;

	return result;
}

// Умножение матриц C = A * B (используется в коде)
Matrix multiply(const Matrix& a, const Matrix& b) {
	Matrix result{};

	for (int j = 0; j < 4; j++) {
		for (int i = 0; i < 4; i++) {
			float sum = 0.0f;
			for (int k = 0; k < 4; k++) {
				sum += a.m[j][k] * b.m[k][i];
			}
			result.m[j][i] = sum;
		}
	}

	return result;
}

VkShaderModule loadShaderModule(const char* path) {
	std::ifstream file(path, std::ios::binary | std::ios::ate);
	if (!file.is_open()) {
		std::cerr << "Failed to open shader file: " << path << "\n";
		return nullptr;
	}
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
	if (vkCreateShaderModule(veekay::app.vk_device, &
	                         info, nullptr, &result) != VK_SUCCESS) {
		return nullptr;
	}

	return result;
}

VulkanBuffer createBuffer(size_t size, void *data, VkBufferUsageFlags usage) {
	VkDevice& device = veekay::app.vk_device;
	VkPhysicalDevice& physical_device = veekay::app.vk_physical_device;
	
	VulkanBuffer result{};

	{
		VkBufferCreateInfo info{
			.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
			.size = size,
			.usage = usage,
			.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
		};

		if (vkCreateBuffer(device, &info, nullptr, &result.buffer) != VK_SUCCESS) {
			std::cerr << "Failed to create Vulkan buffer\n";
			return {};
		}
	}

	{
		VkMemoryRequirements requirements;
		vkGetBufferMemoryRequirements(device, result.buffer, &requirements);

		VkPhysicalDeviceMemoryProperties properties;
		vkGetPhysicalDeviceMemoryProperties(physical_device, &properties);

		const VkMemoryPropertyFlags flags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
		                                     VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

		uint32_t index = UINT_MAX;
		for (uint32_t i = 0; i < properties.memoryTypeCount; ++i) {
			const VkMemoryType& type = properties.memoryTypes[i];

			if ((requirements.memoryTypeBits & (1 << i)) &&
			    (type.propertyFlags & flags) == flags) {
				index = i;
				break;
			}
		}

		if (index == UINT_MAX) {
			std::cerr << "Failed to find required memory type to allocate Vulkan buffer\n";
			return {};
		}

		VkMemoryAllocateInfo info{
			.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
			.allocationSize = requirements.size,
			.memoryTypeIndex = index,
		};

		if (vkAllocateMemory(device, &info, nullptr, &result.memory) != VK_SUCCESS) {
			std::cerr << "Failed to allocate Vulkan buffer memory\n";
			return {};
		}

		if (vkBindBufferMemory(device, result.buffer, result.memory, 0) != VK_SUCCESS) {
			std::cerr << "Failed to bind Vulkan buffer memory\n";
			return {};
		}

		void* device_data;
		vkMapMemory(device, result.memory, 0, requirements.size, 0, &device_data);

		memcpy(device_data, data, size);

		vkUnmapMemory(device, result.memory);
	}

	return result;
}

void destroyBuffer(const VulkanBuffer& buffer) {
	VkDevice& device = veekay::app.vk_device;

	vkFreeMemory(device, buffer.memory, nullptr);
	vkDestroyBuffer(device, buffer.buffer, nullptr);
}

// Генерация геометрии цилиндра (без крышек)
void generateCylinder(std::vector<Vertex>& vertices, std::vector<uint32_t>& indices, float radius, float height, int segments) {
	vertices.clear();
	indices.clear();

	// Вершины
	for (int i = 0; i < segments; ++i) {
		float angle = (float)i / segments * 2.0f * (float)M_PI;
		float x = radius * cosf(angle);
		float z = radius * sinf(angle);

		// Нижняя вершина
		vertices.push_back({{x, -height / 2.0f, z}});
		// Верхняя вершина
		vertices.push_back({{x, height / 2.0f, z}});
	}

	// Индексы для боковой поверхности (Triangle List)
	for (int i = 0; i < segments; ++i) {
		uint32_t i0 = i * 2;
		uint32_t i1 = i * 2 + 1;
		uint32_t i2 = ((i + 1) % segments) * 2;
		uint32_t i3 = ((i + 1) % segments) * 2 + 1;

		// Первый треугольник квада: (i0, i2, i3)
		indices.push_back(i0);
		indices.push_back(i2);
		indices.push_back(i3);

		// Второй треугольник квада: (i0, i3, i1)
		indices.push_back(i0);
		indices.push_back(i3);
		indices.push_back(i1);
	}
}


void initialize() {
	VkDevice& device = veekay::app.vk_device;

	{ // NOTE: Build graphics pipeline
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

		VkPipelineShaderStageCreateInfo stage_infos[2];

		// NOTE: Vertex shader stage
		stage_infos[0] = VkPipelineShaderStageCreateInfo{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.stage = VK_SHADER_STAGE_VERTEX_BIT,
			.module = vertex_shader_module,
			.pName = "main",
		};

		// NOTE: Fragment shader stage
		stage_infos[1] = VkPipelineShaderStageCreateInfo{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.stage = VK_SHADER_STAGE_FRAGMENT_BIT,
			.module = fragment_shader_module,
			.pName = "main",
		};

		// NOTE: How many bytes does a vertex take?
		VkVertexInputBindingDescription buffer_binding{
			.binding = 0,
			.stride = sizeof(Vertex),
			.inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
		};

		// NOTE: Declare vertex attributes
		VkVertexInputAttributeDescription attributes[] = {
			{
				.location = 0, // NOTE: First attribute
				.binding = 0, // NOTE: First vertex buffer
				.format = VK_FORMAT_R32G32B32_SFLOAT, // NOTE: 3-component vector of floats
				.offset = offsetof(Vertex, position), // NOTE: Offset of "position" field in a Vertex struct
			},
		};

		// NOTE: Bring
		VkPipelineVertexInputStateCreateInfo input_state_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
			.vertexBindingDescriptionCount = 1,
			.pVertexBindingDescriptions = &buffer_binding,
			.vertexAttributeDescriptionCount = sizeof(attributes) / sizeof(attributes[0]),
			.pVertexAttributeDescriptions = attributes,
		};

		// NOTE: Every three vertices make up a triangle,
		// so our vertex buffer contains a "list of triangles"
		VkPipelineInputAssemblyStateCreateInfo assembly_state_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
			.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
		};

		// NOTE: Declare clockwise triangle order as front-facing
		// Discard triangles that are facing away
		// Fill triangles, don't draw lines instaed
		VkPipelineRasterizationStateCreateInfo raster_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
			.polygonMode = VK_POLYGON_MODE_FILL,
			.cullMode = VK_CULL_MODE_BACK_BIT,
			.frontFace = VK_FRONT_FACE_CLOCKWISE,
			.lineWidth = 1.0f,
		};

		// NOTE: Use 1 sample per pixel
		VkPipelineMultisampleStateCreateInfo sample_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
			.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
			.sampleShadingEnable = false,
			.minSampleShading = 1.0f,
		};

		VkViewport viewport{
			.x = 0.0f,
			.y = 0.0f,
			.width = static_cast<float>(veekay::app.window_width),
			.height = static_cast<float>(veekay::app.window_height),
			.minDepth = 0.0f,
			.maxDepth = 1.0f,
		};

		VkRect2D scissor{
			.offset = {0, 0},
			.extent = {veekay::app.window_width, veekay::app.window_height},
		};

		// NOTE: Let rasterizer draw on the entire window
		VkPipelineViewportStateCreateInfo viewport_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,

			.viewportCount = 1,
			.pViewports = &viewport,

			.scissorCount = 1,
			.pScissors = &scissor,
		};

		// NOTE: Let rasterizer perform depth-testing and overwrite depth values on condition pass
		VkPipelineDepthStencilStateCreateInfo depth_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
			.depthTestEnable = true,
			.depthWriteEnable = true,
			.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL,
		};

		// NOTE: Let fragment shader write all the color channels
		VkPipelineColorBlendAttachmentState attachment_info{
			.colorWriteMask = VK_COLOR_COMPONENT_R_BIT |
			                  VK_COLOR_COMPONENT_G_BIT |
			                  VK_COLOR_COMPONENT_B_BIT |
			                  VK_COLOR_COMPONENT_A_BIT,
		};

		// NOTE: Let rasterizer just copy resulting pixels onto a buffer, don't blend
		VkPipelineColorBlendStateCreateInfo blend_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,

			.logicOpEnable = false,
			.logicOp = VK_LOGIC_OP_COPY,

			.attachmentCount = 1,
			.pAttachments = &attachment_info
		};

		// NOTE: Declare constant memory region visible to vertex and fragment shaders
		VkPushConstantRange push_constants{
			.stageFlags = VK_SHADER_STAGE_VERTEX_BIT |
			              VK_SHADER_STAGE_FRAGMENT_BIT,
			.size = sizeof(ShaderConstants),
		};

		// NOTE: Declare external data sources, only push constants this time
		VkPipelineLayoutCreateInfo layout_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
			.pushConstantRangeCount = 1,
			.pPushConstantRanges = &push_constants,
		};

		// NOTE: Create pipeline layout
		if (vkCreatePipelineLayout(device, &layout_info,
		                           nullptr, &pipeline_layout) != VK_SUCCESS) {
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

		// NOTE: Create graphics pipeline
		if (vkCreateGraphicsPipelines(device, nullptr,
		                              1, &info, nullptr, &pipeline) != VK_SUCCESS) {
			std::cerr << "Failed to create Vulkan pipeline\n";
			veekay::app.running = false;
			return;
		}
	}

	// ГЕНЕРАЦИЯ БУФЕРОВ ДЛЯ ЦИЛИНДРА
	std::vector<Vertex> vertices;
	std::vector<uint32_t> indices;

	// Генерация цилиндра: радиус 0.5, высота 2.0, ~100 сегментов
	generateCylinder(vertices, indices, 0.5f, 2.0f, CYLINDER_SEGMENTS);
	index_count = (uint32_t)indices.size();

	// Создание буфера вершин
	vertex_buffer = createBuffer(vertices.size() * sizeof(Vertex),
	                             vertices.data(),
	                             VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);

	// Создание буфера индексов
	index_buffer = createBuffer(indices.size() * sizeof(uint32_t),
	                            indices.data(),
	                            VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
}

void shutdown() {
	VkDevice& device = veekay::app.vk_device;

	// NOTE: Destroy resources here, do not cause leaks in your program!
	destroyBuffer(index_buffer);
	destroyBuffer(vertex_buffer);

	vkDestroyPipeline(device, pipeline, nullptr);
	vkDestroyPipelineLayout(device, pipeline_layout, nullptr);
	vkDestroyShaderModule(device, fragment_shader_module, nullptr);
	vkDestroyShaderModule(device, vertex_shader_module, nullptr);
}

void update(double time) {
	// time — аргумент, который предоставляет framework.
	// Обычно это *delta time* (секунды с прошлого кадра), но может быть и абсолют.
	// Мы предполагаем, что это delta time в секундах (типичный случай).
	// Чтобы избежать слишком быстрых эффектов, мы масштабируем время на animation_speed.
	ImGui::Begin("Controls:");
	
	// 1. Управление проекцией
	ImGui::Text("Projection:");
	if (ImGui::RadioButton("Orthographic", current_projection == ProjectionType::ORTHOGRAPHIC)) {
		current_projection = ProjectionType::ORTHOGRAPHIC;
	}
	ImGui::SameLine();
	if (ImGui::RadioButton("Perspective", current_projection == ProjectionType::PERSPECTIVE)) {
		current_projection = ProjectionType::PERSPECTIVE;
	}
	
	if (current_projection == ProjectionType::ORTHOGRAPHIC) {
		ImGui::SliderFloat("Ortho Scale (Half height)", &ortho_scale, 0.5f, 20.0f);
        ImGui::Text("Ortho Z-bounds: [%0.1f, %0.1f]", ORTHO_NEAR, ORTHO_FAR);
	}
	
	ImGui::Separator();

	// 2. Управление Анимацией
	ImGui::Text("Animation Control:");
	
	const char* pause_label = animation_paused ? "Resume (##Pause)" : "Pause (##Pause)";
	if (ImGui::Button(pause_label)) {
		animation_paused = !animation_paused;
	}
	
	ImGui::SameLine();
	
	const char* reverse_label = animation_direction > 0 ? "Reverse (Forward)" : "Reverse (Backward)";
	if (ImGui::Button(reverse_label)) {
		animation_direction *= -1.0f; // Переключаем знак
	}
	
	ImGui::SliderFloat("Animation Speed (time scale)", &animation_speed, 0.0f, 2.0f, "%.3f");
	ImGui::Text("NOTE: This scales the animation time; smaller = slower.");

	ImGui::Separator();
	
	// 3. Сложная траектория и модель
	ImGui::Text("Model Transformation & Trajectory:");
	ImGui::SliderFloat("Trajectory Scale (A)", &trajectory_scale, 0.1f, 8.0f, "%.2f"); 
    ImGui::Text("Trajectory Formula (lemniscate-like): X = A*sin(t), Y = A*sin(t)*cos(t)");
    
	ImGui::SliderFloat("Cylinder Tilt (X-axis)", &cylinder_tilt, 0.0f, 1.5f);
	ImGui::InputFloat3("Translation (Manual)", reinterpret_cast<float*>(&model_position));
	ImGui::SliderFloat("Rotation (Y-axis) Manual (rad)", &model_rotation, 0.0f, 2.0f * (float)M_PI);
	
    ImGui::Checkbox("Spin?", &model_spin);
    if (model_spin) {
        ImGui::SliderFloat("Spin Multiplier (rev/sec)", &spin_speed_multiplier, 0.0f, 1.0f);
    }

	ImGui::ColorEdit3("Color", reinterpret_cast<float*>(&model_color));
	ImGui::End();

	// ЛОГИКА АНИМАЦИИ: Обновление времени
	if (!animation_paused) {
		// --- Главное изменение: current_time уже учитывает animation_speed ---
		// time — delta seconds; animation_speed — множитель времени (time scale).
		current_time += float(time) * animation_direction * animation_speed * 0.05f;
	}

	// t - параметр траектории (используем текущий "внутренний" time напрямую)
	float t = current_time;
	
	// Движение по траектории "восьмерки" / лемниската-подобной
	model_position.x = trajectory_scale * sinf(t);
	model_position.y = trajectory_scale * sinf(t) * cosf(t);
    
    // Для ортографического режима мы хотим, чтобы Z был в диапазоне [ORTHO_NEAR, ORTHO_FAR].
    // Изначально model_position.z был установлен на -5.0f, но если пользователь вводит вручную,
    // мы не перезаписываем его. Здесь — только если текущий Z выбивается, подправим:
    if (current_projection == ProjectionType::ORTHOGRAPHIC) {
        if (model_position.z < ORTHO_NEAR) model_position.z = ORTHO_NEAR + 0.1f;
        if (model_position.z > ORTHO_FAR) model_position.z = ORTHO_FAR - 0.1f;
    }

	// Собственное вращение (spin)
	if (model_spin) {
		// вращение: delta_time * revolutions_per_second * 2pi
		model_rotation += float(time) * animation_direction * spin_speed_multiplier * 2.0f * (float)M_PI; 
	}

	// Ограничение вращения
	model_rotation = fmodf(model_rotation, 2.0f * (float)M_PI);
    if (model_rotation < 0) model_rotation += 2.0f * (float)M_PI;
}

void render(VkCommandBuffer cmd, VkFramebuffer framebuffer) {
	vkResetCommandBuffer(cmd, 0);

	{ // NOTE: Start recording rendering commands
		VkCommandBufferBeginInfo info{
			.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
			.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
		};

		vkBeginCommandBuffer(cmd, &info);
	}

	{ // NOTE: Use current swapchain framebuffer and clear it
		VkClearValue clear_color{.color = {{0.1f, 0.1f, 0.1f, 1.0f}}};
		VkClearValue clear_depth{.depthStencil = {1.0f, 0}};

		VkClearValue clear_values[] = {clear_color, clear_depth};

		VkRenderPassBeginInfo info{
			.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
			.renderPass = veekay::app.vk_render_pass,
			.framebuffer = framebuffer,
			.renderArea = {
				.extent = {
					veekay::app.window_width,
					veekay::app.window_height
				},
			},
			.clearValueCount = 2,
			.pClearValues = clear_values,
		};

		vkCmdBeginRenderPass(cmd, &info, VK_SUBPASS_CONTENTS_INLINE);
	}

	// Обновление констант и отрисовка цилиндра
	{
		// NOTE: Use our new shiny graphics pipeline
		vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);


		// NOTE: Use our vertex buffer
		VkDeviceSize offset = 0;
		vkCmdBindVertexBuffers(cmd, 0, 1, &vertex_buffer.buffer, &offset);

		// NOTE: Use our index buffer
		vkCmdBindIndexBuffer(cmd, index_buffer.buffer, offset, VK_INDEX_TYPE_UINT32);

		// ПОСТРОЕНИЕ МОДЕЛЬНОЙ МАТРИЦЫ: M = R_X(Tilt) * R_Y * T 
        // Порядок применения: Перемещение -> Вращение (вокруг Y) -> Наклон (вокруг X)
        // Порядок умножения матриц (для V_world = M * V_model): M = Tilt * Rotation_Y * Translation
		
		Matrix translation_matrix = translation(model_position);
		Matrix rotation_y = rotation({0.0f, 1.0f, 0.0f}, model_rotation);
		Matrix rotation_x_tilt = rotation({1.0f, 0.0f, 0.0f}, cylinder_tilt);
		
		// Общая трансформация = Rotation_X_Tilt * (Rotation_Y * Translation)
		Matrix transform_rot_y_trans = multiply(rotation_y, translation_matrix);
		Matrix final_transform = multiply(rotation_x_tilt, transform_rot_y_trans);


		ShaderConstants constants{
			.projection = projection(
				camera_fov,
				float(veekay::app.window_width) / float(veekay::app.window_height),
				camera_near_plane, camera_far_plane),

			// Применяем финальную трансформацию
			.transform = final_transform,

			.color = model_color,
		};

		// NOTE: Update constant memory with new shader constants
		vkCmdPushConstants(cmd, pipeline_layout,
		                   VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
		                   0, sizeof(ShaderConstants), &constants);

		// NOTE: Draw the cylinder using the index count
		vkCmdDrawIndexed(cmd, index_count, 1, 0, 0, 0);
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
