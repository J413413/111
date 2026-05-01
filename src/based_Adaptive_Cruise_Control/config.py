"""
配置文件 - 自适应巡航控制强化学习项目
"""

# 车辆物理参数
TARGET_SPEED = 25.0  # 目标巡航速度 (m/s)
SAFETY_DISTANCE = 15.0  # 安全距离 (m)
MAX_ACCELERATION = 2.0  # 最大加速度 (m/s²)
MAX_DECELERATION = -3.0  # 最大减速度 (m/s²)
DT = 0.1  # 时间步长 (s)
MAX_SPEED = 35.0  # 最大允许速度 (m/s)
MIN_SPEED = 0.0  # 最小允许速度 (m/s)

# 环境参数
EPISODE_LENGTH = 1000  # 每个回合的最大步数
OBSERVATION_DIM = 5  # 观测空间维度
ACTION_DIM = 1  # 动作空间维度

# 训练参数
TRAINING_TIMESTEPS = 1000000  # 总训练步数
SAVE_FREQUENCY = 100000  # 模型保存频率
LOG_FREQUENCY = 1000  # 日志记录频率
BATCH_SIZE = 64  # 批次大小
GAMMA = 0.99  # 折扣因子
LEARNING_RATE = 3e-4  # 学习率
N_EPOCHS = 10  # PPO  epochs
CLIP_RANGE = 0.2  # PPO 裁剪范围

# 路径设置
MODEL_DIR = "models"  # 模型保存目录
LOG_DIR = "logs"  # 日志保存目录

# 测试参数
TEST_EPISODES = 10  # 测试回合数
TEST_RENDER = True  # 是否渲染测试过程