# FLAME GPU 2 康威生命游戏模拟 (Conway's Game of Life Simulation)

## 项目概述

本项目使用 PyFLAMEGPU 2 库实现了一个经典的元胞自动机模型——康威生命游戏（Conway's Game of Life）。它展示了如何利用 FLAME GPU 2 框架的强大功能，在 GPU 上高效地进行大规模基于智能体（Agent-Based）的模拟。主要目的是演示 FLAME GPU 2 的基本概念，包括模型定义、环境设置、智能体行为、消息传递和执行流程控制。

## 文件结构

*   `model.py`: Python 脚本，用于定义 FLAME GPU 2 模型的结构，包括环境、智能体、消息、函数和执行层级，并负责初始化和运行模拟。
*   `cuda.py`: 包含用 C++ 编写并在运行时编译 (RTC) 的 CUDA Agent 函数的 Python 字符串。这些函数定义了每个智能体（细胞）在模拟中每一步的具体行为。
*   `README.md`: 本文件，提供项目介绍、代码逻辑解释和设计思路。

## 核心逻辑与设计思路 (`model.py` & `cuda.py`)

### 1. 框架选择：FLAME GPU 2

*   **原因**：选择 FLAME GPU 2 是因为它是一个专为在 GPU 上执行大规模 Agent-Based Modeling (ABM) 而设计的框架。它通过 CUDA 将计算密集型的 Agent 函数并行化，极大地加速了模拟过程，特别适合像生命游戏这样包含大量简单、局部交互单元的系统。PyFLAMEGPU 2 提供了 Python 接口，使得模型定义和模拟控制更加便捷。

### 2. 模型定义 (`model.py`)

模型定义遵循 FLAME GPU 2 的标准结构，将模拟的不同方面模块化：

*   **`create_model()`**:
    *   **逻辑**: 创建一个模型实例并命名为 "Game of life"。
    *   **原因**: 这是定义任何 FLAME GPU 模拟的第一步，为整个模拟提供一个容器。

*   **`define_environment()`**:
    *   **逻辑**: 定义模型运行的全局环境属性。这里设置了 `SQRT_AGENT_COUNT`（用于计算二维网格的边长）和 `AGENT_COUNT`（总智能体数量），以及 `repulse` 和 `radius` (虽然在此版生命游戏中未使用，但展示了定义环境参数的方法)。
    *   **原因**: 环境属性是全局常量或可在模拟步骤之间修改的变量，它们影响所有 Agent 或模拟整体行为，但不属于任何特定 Agent。将网格尺寸等参数放在环境中，便于全局访问和配置。

*   **`define_messages()`**:
    *   **逻辑**: 定义了一个名为 `is_alive_message` 的 `MessageArray2D` 类型的消息列表。这种消息类型特别适合网格状空间中的邻居通信。消息包含一个 `is_alive` 变量（`char` 类型，节省��间）。消息列表的维度设置为 `SQRT_AGENT_COUNT x SQRT_AGENT_COUNT`，与 Agent 所在的逻辑网格对应。
    *   **原因**: 消息传递是 FLAME GPU 中 Agent 交互的核心机制。`MessageArray2D` 允许 Agent 将其状态（是否存活）发送到其在逻辑网格中的对应位置。其他 Agent 可以通过查询其邻域内的消息来获取邻居的状态，而无需直接访问其他 Agent 的内部变量。这对于 GPU 并行化至关重要，避免了读写冲突和复杂的同步问题。

*   **`define_agents()`**:
    *   **逻辑**: 定义了名为 "cell" 的 Agent 类型。
        *   **变量**: 每个 Cell Agent 拥有 `pos` (一个包含两个 `unsigned int` 的数组，表示其在网格中的逻辑坐标 x, y)，`is_alive` (一个 `unsigned int`，表示细胞存活状态，0 或 1)，以及仅在启用可视化时存在的 `x`, `y` (浮点数，用于在可视化空间中定位)。
        *   **函数**: 绑定了两个 Agent 函数：`output` 和 `update`。这两个函数是通过 RTC (Run-Time Compilation) 从 `cuda.py` 文件中的 C++ 代码字符串加载的。`output` 函数被指定为输出到 `is_alive_message`，而 `update` 函数则从 `is_alive_message` 读取输入。
    *   **原因**: Agent 是模拟的基本单元。定义其状态变量 (`pos`, `is_alive`) 和行为函数 (`output`, `update`) 是 ABM 的核心。将可视化相关的变量 (`x`, `y`) 条件性地包含进来，可以减少非可视化运行时 Agent 的内存占用。使用 RTC 加载 C++ Agent 函数是为了获得最佳性能，因为这些函数将在 GPU 上由数千甚至数百万 Agent 并行执行，C++/CUDA 通常比纯 Python 更高效。

*   **`define_execution_order()`**:
    *   **逻辑**: 定义了两个执行层（Layer）。第一层执行所有 "cell" Agent 的 `output` 函数。第二层执行所有 "cell" Agent 的 `update` 函数。
    *   **原因**: 层级定义了在一个模拟步骤（Step）内 Agent 函数的执行顺序。在生命游戏中，所有细胞必须**首先**根据其当前状态广播自己是否存活（`output` 阶段），然后所有细胞再**同时**根据邻居的状态（从消息中读取）更新自己的下一状态（`update` 阶段）。如果 `output` 和 `update` 在同一层，或者顺序颠倒，可能会导致 Agent 基于邻居的 *新* 状态（而非上一轮的状态）进行更新，从而产生错误或不确定的结果。分层确保了状态更新的同步性和确定性，符合元胞自动机的规则。

### 3. CUDA C++ 核心 (`cuda.py`)

这些是在 GPU 上并行执行的 Agent 函数的 C++ 实现。

*   **`output` 函数**:
    *   **逻辑**: 每个 Agent 读取自己的 `is_alive` 状态和 `pos` 坐标。然后，它将 `is_alive` 状态（转换为 `char` 以节省消息内存）写入到 `is_alive_message` 消息列表中的 `(pos[0], pos[1])` 索引位置。
    *   **原因**: 这个函数实现了状态广播。每个细胞将其当前状态发布到共享的消息网格中，供邻居在下一个阶段（`update`）读取。

*   **`update` 函数**:
    *   **逻辑**:
        1.  获取 Agent 自身的坐标 `my_x`, `my_y`。
        2.  使用 `FLAMEGPU->message_in.wrap(my_x, my_y)` 迭代器访问其在 `is_alive_message` 网格中的 3x3 摩尔邻域（Moore neighborhood）的消息（`wrap` 表示使用环绕边界条件，即网格边缘连接到对侧）。
        3.  计算存活邻居的数量 `living_neighbours`。
        4.  获取 Agent 当前的 `is_alive` 状态。
        5.  根据康威生命游戏的标准规则（存活细胞在邻居 < 2 或 > 3 时死亡，死亡细胞在邻居 == 3 时复活）计算下一状态。
        6.  使用 `FLAMEGPU->setVariable<unsigned int>("is_alive", ...)` 更新 Agent 自身的 `is_alive` 变量。
    *   **原因**: 这个函数实现了生命游戏的核心规则。它从消息系统中读取邻居信息（解耦了 Agent 间的直接依赖），应用规则进行状态转换。这是计算密集的部分，因此用 C++/CUDA 编写以在 GPU 上高效执行。

### 4. 初始化与执行 (`initialise_simulation`, `if __name__ == "__main__":`)

*   **逻辑**:
    *   `initialise_simulation`: 按照定义的顺序调用模型构建函数，创建 `CUDASimulation` 对象，初始化模拟器。如果启用了可视化 (`pyflamegpu.VISUALISATION`)，则配置相机、Agent 视觉样式（立方体，颜色根据 `is_alive` 状态变化）并激活窗口。如果没有提供输入模型文件，则程序化地生成初始 Agent 种群：根据 `AGENT_COUNT` 创建 Agent，将它们放置在网格上，并以约 40% 的概率随机设置为存活状态。最后，将生成的种群数据设置到模拟器中，并调用 `cudaSimulation.simulate()` 开始运行。
    *   `if __name__ == "__main__":`: 作为脚本入口点，调用 `initialise_simulation`，并简单计时模拟的总执行时间。
*   **原因**: 这部分代码负责将定义好的模型蓝图实例化为具体的模拟，并启动执行。可视化配置使得可以直接观察模拟过程。程序化生成初始种群提供了一个方便的默认启动方式，无需手动创建输入文件。计时代码用于基本的性能评估。

## 运行方式

1.  确保已安装 PyFLAMEGPU 2 及其依赖项（包括兼容的 CUDA 工具包）。
2.  在命令行中运行 Python 脚本：
    ```bash
    python model.py
    ```
3.  如果 PyFLAMEGPU 编译时开启了可视化，并且代码中 `pyflamegpu.VISUALISATION` 为 `True`，则会弹出一个可视化窗口。

## 可视化

*   当 `pyflamegpu.VISUALISATION` 为 `True` 时，模拟会以 3D 形式展示。
*   每个细胞表示为一个立方体。
*   立方体的颜色表示细胞状态：白色代表存活 (is_alive=1)，黑色代表死亡 (is_alive=0)。 (注：颜色映射在 `initialise_simulation` 中定义)
*   可以通过鼠标和键盘与可视化窗口交互（缩放、平移、旋转）。

## 未来参考价值

这个项目展示了使用 PyFLAMEGPU 2 构建 Agent-Based Simulation 的典型流程和关键设计模式，可供未来开发参考：

1.  **模块化设计**: 将模型定义分解为环境、Agent、消息、函数和层级，结构清晰。
2.  **Python + RTC C++**: 利用 Python 的易用性进行模型结构定义和模拟控制，同时使用 RTC C++ 编写性能关键的 Agent 逻辑，实现开发效率和运行性能的平衡。
3.  **基于消息的交互**: 使用专门的消息类型（如 `MessageArray2D`）进行 Agent 间的通信，这是 FLAME GPU 实现高效并行计算和解耦 Agent 的核心机制。未来项目可根据交互模式选择不同的消息类型。
4.  **执行顺序控制**: 通过分层精确控制 Agent 函数的执行顺序，确保模拟逻辑的正确性和确定性。
5.  **环境参数**: 使用环境属性来管理全局设置。
6.  **条件化可视化**: 将可视化代码与核心模拟逻辑分离，便于在不同模式下运行。
7.  **程序化种群生成**: 提供了一种便捷的初始化模拟状态的方法。

通过理解本项目的结构和设计原因，您可以将这些模式和技术应用到更复杂的 Agent-Based Simulation 项目中。
