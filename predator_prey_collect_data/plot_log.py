import json
import matplotlib.pyplot as plt

def plot_agent_counts(log_file="log.json"):
    """
    读取FLAME GPU生成的log.json文件，并绘制智能体数量随时间步变化的曲线图。

    Args:
        log_file (str): FLAME GPU log文件的路径，默认为"log.json"。
    """
    try:
        with open(log_file, 'r') as f:
            log_data = json.load(f) # 读取 JSON 文件内容 [1, 2, 5, 6]

        steps = []
        prey_counts = []
        predator_counts = []
        grass_counts = []

        # 提取每个时间步的智能体数量
        for step_data in log_data["steps"]:
            steps.append(step_data["step_index"])
            prey_counts.append(step_data["agents"]["prey"]["default"]["count"])
            predator_counts.append(step_data["agents"]["predator"]["default"]["count"])
            grass_counts.append(step_data["agents"]["grass"]["default"]["count"])

        # 使用 matplotlib 绘图
        plt.figure(figsize=(10, 6)) # 设置图的大小

        plt.plot(steps, prey_counts, label="Prey", marker='o', linestyle='-') # 绘制猎物数量曲线 [7, 8, 9, 10]
        plt.plot(steps, predator_counts, label="Predator", marker='x', linestyle='--') # 绘制捕食者数量曲线
        plt.plot(steps, grass_counts, label="Grass", marker='s', linestyle='-.') # 绘制草地数量曲线

        # 添加图例和标签
        plt.xlabel("Simulation Step")
        plt.ylabel("Agent Count")
        plt.title("Agent Count Over Simulation Steps")
        plt.legend() # 显示图例

        # 添加网格线
        plt.grid(True)

        # 显示图表
        plt.show()

    except FileNotFoundError:
        print(f"Error: Log file '{log_file}' not found.")
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{log_file}'. Please ensure it's a valid JSON file.")
    except KeyError as e:
        print(f"Error: Missing key in JSON data - {e}. Please check the structure of the log file.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    plot_agent_counts("log.json") # 调用函数进行绘图
