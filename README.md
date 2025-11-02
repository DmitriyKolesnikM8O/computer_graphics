# 🌋 Veekay

## Getting started

You need C++ compiler, Vulkan SDK and CMake installed before you can build this project.

This project uses C++20 standard and thus requires either of those compilers:
- GCC 10.X
- Clang 10
- Microsoft Visual Studio 2019

Veekay is officially tested on *Windows* and *GNU/Linux platforms*, no *macOS* support yet.
If you have a working macOS solution of this code, consider submitting a PR so others
can build this example code without a hassle!

<ins>**1. Downloading the repository**</ins>

Start by cloning the repository with `git clone --depth 1 https://github.com/vladeemerr/veekay`

This repository does not contain any submodules, it utilizes CMake's `FetchContent` feature instead.

<ins>**2. Configuring the project**</ins>

Run either one of the CMake lines to download dependencies and configure the project:

```bash
cmake --preset debug      # for GNU/Linux (GCC/Clang)
cmake --preset msvc-debug # for Windows (Visual Studio 2019)
```

If you wish to build in `release` mode, change `debug` to `release`.

If changes are made (added/removed files), or if you want to regenerate project files, rerun the command above.

<ins>**3. Building**</ins>

To build the project, use the line below. You are most likely using `debug` preset, so
the directory that will eventually contain your build files is named `build-debug`.

Likewise for `release` that directory will be named `build-release`

Run one those commands, depending on which preset you chose:

```bash
cmake --build build-debug --parallel # for debug
cmake --build build-release --parallel # for release
```

## Project structure

### Overview

Veekay consists of two parts: library and application

* `source` directory contains library code
* `testbed` directory contains application code

Library code contains most of the boilerplate for GLFW, Vulkan and ImGui initialization.
Veekay library also takes care of managing swapchain and giving you relevant
`VkCommandBuffer` and `VkFramebuffer` for you to submit/render to.

However, the majority of your work will happen in `testbed`.
This is where you will write most of your application code.
It is already linked with Veekay library and contains its own `CMakeLists.txt`
build recipe for you to modify.

### Application code

`veekay.hpp` header exposes library functionality through a set of callbacks
(`init`, `shutdown`, `update`, `render`) and global variable `app`.

Look for `testbed/main.cpp`, this is where you start.

`veekay::Application` contains important data like window size, `VkDevice`,
`VkPhysicalDevice` and `VkRenderPass` (associated with a swapchain).

So, say you want to create a `VkBuffer`. This is how you would do it:

```c++
// You fill info struct before calling vkCreateXXX
vkCreateBuffer(veekay::app.vk_device, &info, nullptr, &vertex_buffer);
```

Notice the `veekay::app.vk_device`, `veekay` is a library namespace,
`app` is a global state variable provided by Veekay and `vk_device` is
a `VkDevice` contained in `app` variable.

### Running

`build-xxx/testbed` will contain the executable after successful build

**Make sure your working directory is set to the project root!**
Project root is where this README file resides. Otherwise, the
code responsible for loading shaders from files will fail, because relative paths are used.

### Compiling shaders

`testbed/CMakeLists.txt` has build recipe for compiling shader files
along with an application. Look for a comment in this file to see
how to compile your shaders.


### ЛАБА
Мы создали 3D-сцену с несколькими объектами и сложным, настраиваемым освещением. Мы можем свободно летать по этой сцене с помощью камеры, управляемой мышью и клавиатурой, и в реальном времени менять цвет, положение и тип источников света через UI-панели.

- main.cpp (Главный файл, Командный центр на CPU)
Ключевые функции:
    - initialize(): Строительство. Один раз при запуске настраивает весь рендеринг: загружает шейдеры, описывает формат вершин, создает буферы для данных, настраивает конвейер Vulkan.
    - update(): Обновление каждый кадр. Реагирует на ввод с клавиатуры/мыши для движения камеры. Рисует UI-панели (ImGui). Собирает все данные (матрицы, цвета, позиции света) и копирует их в специальные буферы, видимые для GPU.
    - render(): Отправка команд на отрисовку. Не рисует сама, а записывает в командный буфер последовательность приказов для GPU: "Начать рисовать, использовать вот эти шейдеры, вот эти данные, нарисовать вот этот объект".
    - shutdown(): Уборка. В конце освобождает всю выделенную память и уничтожает все созданные объекты Vulkan.
  
- shaders/shader.vert (Вершинный шейдер, работает на GPU)
Выполняется для каждой вершины (угла) 3D-модели.
Задача: Взять локальную координату вершины (например, {-0.5, -0.5, -0.5} для угла куба) и, используя матрицы model, view, projection, вычислить ее финальную позицию на 2D-экране. Эту позицию он записывает в gl_Position.
Также он передает дальше (во фрагментный шейдер) обработанную позицию и нормаль вершины.

- shaders/shader.frag (Фрагментный шейдер, работает на GPU)
Выполняется для каждого пикселя, из которых состоит объект на экране.
Задача: Определить финальный цвет этого пикселя. Это сердце нашей лабораторной. Здесь реализована вся логика освещения. Он берет нормаль поверхности, позицию пикселя, данные о материале (цвет, блеск) и данные обо всех источниках света, чтобы по формуле Блинн-Фонга вычислить итоговый цвет.

- source/graphics.cpp и include/veekay/types.hpp (Вспомогательные файлы)
    - types.hpp: Определяет математические типы (vec3, mat4) и операции над ними (умножение матриц, нормализация векторов, look_at).
    - graphics.cpp: Часть каркаса, которая помогает упростить работу с Vulkan, например, создание буферов.


РАЗБОР ПО ПУНКТАМ:
1. Модель освещения Блинн-Фонга
   Где: shader.frag, функция calculate_blinn_phong
   Как: Функция принимает нормаль N, вектор к камере V, вектор к свету L и другие параметры. Ключевая строка vec3 H = normalize(V + L); вычисляет вектор полупути. Затем pow(max(dot(N, H), 0.0), model.shininess); вычисляет блик. Эта функция используется для всех типов света.
2. Разные типы источников света
   Где: shader.frag, функция main().
   Как:
    - Ambient (рассеянный): vec3 color = pc.ambient_color * model.albedo_color;. Просто умножение цвета окружения на цвет объекта.
    - Directional (направленный): color += calculate_blinn_phong(...) вызывается один раз с постоянным направлением света (pc.directional_dir).
    - Point (точечные): for (uint i = 0; i < lights.point_light_count; ++i) — запускается цикл. Внутри него для каждого источника из массива lights.point_lights вычисляется направление L, затухание по закону обратных квадратов (attenuation = 1.0 / (constant + linear * dist + ...)), и вызывается calculate_blinn_phong.
    - Spot (прожектор): color += calculate_spot_light(N, V);. Вызывается специальная функция, которая, помимо calculate_blinn_phong, дополнительно проверяет, попадает ли пиксель в конус света, и вычисляет интенсивность с учетом плавных краев.
3. Камера и управление
   Где: main.cpp, структура Camera и функция update().
   Как:
    - Режим трансформации (Вариант 1): В функции Camera::view() блок if (!is_look_at) вычисляет матрицу вида через комбинацию обратных поворотов и переноса. Вращение (camera.rotation) меняется в update() на основе движения мыши.
    - Режим Look-At (Доп. задание): В Camera::view() блок if (is_look_at) вызывает veekay::mat4::look_at, используя camera.position и camera.target.
    - Перемещение: В update() блок if (veekay::input::keyboard::isKeyDown(...)) меняет camera.position при нажатии на W/A/S/D/Q/Z.
4. Передача данных в GPU
   Где: Везде.
   Как:
    - Данные вершин (позиция, нормаль): Загружаются в vertex_buffer в initialize(). GPU получает их через in переменные в shader.vert.
    - Материалы (цвет, блеск) и матрица модели: Копируются в model_uniforms_buffer в update(). Шейдеры получают их через uniform ModelUniforms. Так как для каждого объекта они свои, используется Dynamic UBO (передача смещения dyn_offset в vkCmdBindDescriptorSets).
    - Точечные источники: Копируются в light_ssbo_buffer в update(). Шейдер получает их через SSBO (readonly buffer LightSSBO). Это позволяет передать целый массив данных.
    - Остальные источники света и позиция камеры: Собираются в структуру Push в render() и передаются через Push Constants (vkCmdPushConstants). Это самый быстрый способ для небольшого объема данных.
5. Пользовательский интерфейс (UI)
   Где: main.cpp, функция update().
   Как: С помощью библиотеки ImGui. Команды ImGui::Begin(), ImGui::SliderFloat3(), ImGui::ColorEdit3(), ImGui::Checkbox() создают окна и виджеты, которые напрямую читают и изменяют значения C++ переменных (например, ambient_light.color, camera.is_look_at).