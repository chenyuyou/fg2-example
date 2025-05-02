# README.md

## FLAME GPU 2D 空间交互模拟

### 概述

本项目使用 FLAME GPU 2 (通过其 Python 绑定 `pyflamegpu`) 实现了一个简单的二维空间代理（agent）模拟。在此模拟中，大量代理在二维环境中移动，并根据与其他代理的接近程度相互排斥。该项目演示了 FLAME GPU 的核心概念，包括模型定义、环境属性、代理行为、空间消息传递和 CUDA 代理函数。

### 文件结构

*   `model.py`: Python 脚本，用于定义 FLAME GPU 模型、环境、消息、代理、执行顺序，并初始化和运行模拟。
*   `cuda.py`: 包含 CUDA C++ 编写的代理函数（Agent Function）字符串。这些函数定义了单个代理在 GPU 上执行的具体行为逻辑。
*   `README.md`: 本文档，提供代码的详细解释。

### 代码详解

#### `cuda.py` - CUDA 代理函数

该文件包含两个核心的 CUDA C++ 字符串，它们定义了代理的行为。FLAME GPU 会在运行时编译这些字符串为可在 GPU 上执行的 CUDA 核函数。

1.  **`output_message` 函数**
    *   **签名**:
        ```cuda
        FLAMEGPU_AGENT_FUNCTION(output_message, flamegpu::MessageNone, flamegpu::MessageSpatial2D) { ... }
        ```
        *   定义了一个名为 `output_message` 的代理函数。
        *   `flamegpu::MessageNone`: 表示此函数不读取任何类型的消息。
        *   `flamegpu::MessageSpatial2D`: 表示此函数将输出一个 `Spatial2D` 类型的消息（在此例中为 "location" 消息）。
    *   **目的**: 每个代理执行此函数以广播其当前状态（ID 和位置）给其他代理。这是信息共享的第一步。
    *   **实现逻辑**:
        *   `FLAMEGPU->message_out.setVariable<flamegpu::id_t>("id", FLAMEGPU->getID());`: 获取当前代理的唯一 ID，并将其设置到输出消息的 "id" 变量中。这使得接收消息的代理可以识别消息来源，避免自我交互。
        *   `FLAMEGPU->message_out.setLocation(...)`: 将当前代理的 "x" 和 "y" 变量（即其坐标）设置到输出消息的空间位置。`Spatial2D` 消息系统会利用这个位置信息进行高效的空间索引和查询。
        *   `return flamegpu::ALIVE;`: 表示该代理在当前模拟步骤后仍然存活。

2.  **`input_message` 函数**
    *   **签名**:
        ```cuda
        FLAMEGPU_AGENT_FUNCTION(input_message, flamegpu::MessageSpatial2D, flamegpu::MessageNone) { ... }
        ```
        *   定义了一个名为 `input_message` 的代理函数。
        *   `flamegpu::MessageSpatial2D`: 表示此函数将读取 `Spatial2D` 类型的消息（"location" 消息）。
        *   `flamegpu::MessageNone`: 表示此函数不输出任何消息。
    *   **目的**: 每个代理执行此函数以感知其周围环境（读取附近代理的位置消息），并根据这些信息计算自身的移动。这是代理交互和行为决策的核心。
    *   **实现逻辑**:
        *   获取自身 ID (`ID`) 以便后续过滤掉自己的消息。
        *   从环境和消息定义中获取排斥力因子 (`REPULSE_FACTOR`) 和消息交互半径 (`RADIUS`)。
        *   初始化合力分量 `fx`, `fy`。
        *   获取代理当前的 `x1`, `y1` 坐标。
        *   `for (const auto &message : FLAMEGPU->message_in(x1, y1))`: **关键的空间查询**。高效迭代处理以 `(x1, y1)` 为中心、`RADIUS` 为半径区域内的 "location" 消息。
        *   过滤掉自身消息 (`message.getVariable<flamegpu::id_t>("id") != ID`)。
        *   获取消息来源代理的位置 `x2`, `y2`。
        *   计算距离 (`separation`)。
        *   只处理交互半径内且非重合的代理 (`separation < RADIUS && separation > 0.0f`)。
        *   计算排斥力大小 `k` (基于正弦函数和距离)。
        *   获取方向向量并计算力的分量，累加到 `fx`, `fy`。
        *   计算平均作用力 (`fx /= count > 0 ? count : 1; fy /= count > 0 ? count : 1;`)。
        *   根据平均力更新代理位置 (`x = x1 + fx`, `y = y1 + fy`)。
        *   计算并存储位置变化幅度 (`drift`)。
        *   `return flamegpu::ALIVE;`: 代理继续存活。

#### `model.py` - Python 模型设置脚本

此脚本使用 `pyflamegpu` 库来配置和运行整个模拟。

1.  **`create_model()`**:
    *   **目的**: 创建模型描述对象 (`pyflamegpu.ModelDescription`)。
    *   实例化模型并命名。

2.  **`define_environment()`**:
    *   **目的**: 定义全局环境变量。
    *   获取环境对象 (`model.Environment()`)。
    *   定义 `AGENT_COUNT`, `ENV_WIDTH`, `repulse` 等属性。

3.  **`define_messages()`**:
    *   **目的**: 定义代理间通信的消息类型。
    *   创建 `Spatial2D` 消息 "location"。
    *   添加消息变量 "id"。
    *   设置交互半径 (`setRadius`) 和空间边界 (`setMin`, `setMax`)。

4.  **`define_agents()`**:
    *   **目的**: 定义代理类型。
    *   创建代理 "point" (`model.newAgent("point")`)。
    *   定义代理变量 "x", "y", "drift"。
    *   创建 RTC 代理函数 `output_message` 和 `input_message`，关联 `cuda.py` 中的代码字符串。
    *   指定函数的消息输入 (`setMessageInput`) 和输出 (`setMessageOutput`)。
    *   **关键连接**: 将 CUDA 代码逻辑与代理类型关联。

5.  **`define_execution_order()`**:
    *   **目的**: 定义模拟步骤中函数的执行顺序（层）。
    *   创建第一层，添加 "point" 代理的 `output_message` 函数。
    *   创建第二层，添加 "point" 代理的 `input_message` 函数。
    *   **设计原因**: 确保先输出旧状态，再根据接收到的信息计算新状态，避免读写冲突。

6.  **`initialise_simulation(seed)`**:
    *   **目的**: 整合定义，设置模拟器，处理可视化，初始化种群，启动模拟。
    *   调用前面的 `define_...` 函数。
    *   创建 CUDA 模拟器 (`pyflamegpu.CUDASimulation`)。
    *   **(可选) 可视化设置**:
        *   检查 `pyflamegpu.VISUALISATION` 标志。
        *   设置相机、渲染代理（形状、大小）、绘制边界线框。
        *   激活可视化窗口 (`m_vis.activate()`)。
    *   初始化模拟器 (`cudaSimulation.initialise(sys.argv)`), 可处理命令行参数。
    *   **种群初始化**:
        *   检查是否有输入文件 (`SimulationConfig().input_file`)。
        *   若无，则创建代理向量 (`pyflamegpu.AgentVector`)，随机设置初始位置。
        *   加载种群数据 (`cudaSimulation.setPopulationData(population)`)。
    *   启动模拟 (`cudaSimulation.simulate()`)。
    *   **(可选, 已注释)** 保持可视化窗口 (`m_vis.join()`)。

7.  **`if __name__ == "__main__":`**:
    *   脚本入口点。
    *   计时 (`time.time()`)。
    *   调用 `initialise_simulation()` 启动模拟。
    *   打印执行时间。
    *   退出。

### 如何运行

1.  确保已安装 `pyflamegpu` 及其依赖项（包括 CUDA 工具包和兼容的 GPU 驱动）。
2.  在终端中运行 Python 脚本：
    ```bash
    python model.py
    ```
3.  如果 `pyflamegpu` 编译时启用了可视化，将弹出一个可视化窗口显示模拟过程。

### 依赖项

*   `pyflamegpu`: FLAME GPU 2 的 Python 绑定。
*   Python 3.x
*   CUDA Toolkit (版本需与 `pyflamegpu` 兼容)
*   NVIDIA GPU (具有适当的计算能力)
