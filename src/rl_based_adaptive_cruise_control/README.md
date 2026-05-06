# 🚗 RL-Based-Adaptive-Cruise-Control

本项目旨在利用**深度强化学习**（Deep Reinforcement Learning, DRL）技术，特别是 **PPO (Proximal Policy Optimization)** 算法，来训练一个智能体，实现**自适应巡航控制**（Adaptive Cruise Control, ACC）功能。

传统的 ACC 系统通常基于 PID 或 MPC（模型预测控制），依赖于精确的物理模型和繁琐的参数调优。本项目探索如何通过端到端的强化学习方法，让智能体在与环境的交互中自主学习最优的跟车策略，以实现安全、舒适且高效的自动驾驶体验。

---

## ✨ 功能特点

- 🤖 **强化学习驱动**：使用 `Stable-Baselines3` 库实现 PPO 算法，处理连续动作空间。
- 🚦 **自定义仿真环境**：基于 `Gym` 框架构建了简化的车辆跟驰环境，模拟前车随机加减速场景。
- ⚖️ **多目标优化**：奖励函数综合考虑了**速度跟踪**（保持设定速度）、**安全距离**（防止碰撞）和**乘坐舒适度**（避免急加减速）。
- 📊 **可视化支持**：包含简单的状态输出，方便调试和观察训练过程。

---

## 🛠️ 环境依赖

确保您的系统已安装 Python 3.7 或更高版本。

本项目主要依赖以下库：

- `gym` (>=0.21.0): 用于构建强化学习环境接口。
- `stable-baselines3` (>=1.6.0): 提供高性能的强化学习算法实现。
- `numpy`: 用于数值计算。
- `shimmy`: 用于兼容旧版 Gym 接口。

### 安装步骤

1. 克隆本项目到本地：
   ```bash
   git clone https://github.com/YOUR_USERNAME/RL-Based-Adaptive-Cruise-Control.git
   cd RL-Based-Adaptive-Cruise-Control
   ```

2. 安装所需依赖：
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 快速开始

### 训练模型

运行以下命令开始训练 ACC 智能体：

```bash
python train.py
```

训练过程中，模型会定期保存到 `models/` 目录下。

### 测试模型

使用训练好的模型进行测试：

```bash
python test.py --model models/best_model.zip
```

### 可视化结果

训练完成后，可以使用以下命令生成结果可视化：

```bash
python visualize.py --model models/best_model.zip
```

---

## 📁 项目结构

```
RL-Based-Adaptive-Cruise-Control/
├── acc_env/                  # 自适应巡航控制环境
│   ├── __init__.py
│   └── acc_env.py           # 环境定义
├── models/                  # 保存训练好的模型
├── utils/                   # 工具函数
│   ├── __init__.py
│   └── reward_functions.py  # 奖励函数定义
├── config.py                # 配置文件
├── train.py                 # 训练脚本
├── test.py                  # 测试脚本
├── visualize.py             # 可视化脚本
├── requirements.txt         # 依赖列表
└── README.md                # 项目说明
```

---

## 📝 配置说明

主要配置参数位于 `config.py` 文件中：

- `TARGET_SPEED`: 目标巡航速度 (m/s)
- `SAFETY_DISTANCE`: 安全距离 (m)
- `MAX_ACCELERATION`: 最大加速度 (m/s²)
- `MAX_DECELERATION`: 最大减速度 (m/s²)
- `DT`: 时间步长 (s)
- `EPISODE_LENGTH`: 每个回合的最大步数
- `TRAINING_TIMESTEPS`: 总训练步数
- `SAVE_FREQUENCY`: 模型保存频率

---

## 🔧 自定义环境

ACC 环境模拟了一个简化的车辆跟驰场景：

- **状态空间**：包含自车速度、前车速度、两车相对距离等信息。
- **动作空间**：连续动作，表示加速度指令。
- **奖励函数**：综合考虑速度跟踪误差、安全距离保持和加速度平滑性。

如需修改环境参数或奖励函数，请参考 `acc_env/acc_env.py` 和 `utils/reward_functions.py` 文件。

---

## 📈 结果分析

训练完成后，可以通过以下指标评估模型性能：

1. **速度跟踪误差**：实际速度与目标速度的偏差
2. **安全距离保持**：与前车的距离是否始终保持在安全范围内
3. **乘坐舒适度**：加速度变化的平滑程度
4. **燃油效率**：平均加速度和速度的关系

---

## 🤝 贡献指南

欢迎提交 Issue 和 Pull Request 来改进这个项目！

---

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件。

---

## 🙏 致谢

感谢以下项目和资源的启发：

- [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3)
- [OpenAI Gym](https://github.com/openai/gym)
- [CARLA Simulator](https://github.com/carla-simulator/carla)