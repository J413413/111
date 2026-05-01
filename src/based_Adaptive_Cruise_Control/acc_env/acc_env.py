"""
自适应巡航控制环境 - 基于OpenAI Gym框架
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register

from utils.reward_functions import calculate_reward

class ACCEnv(gym.Env):
    """
    自适应巡航控制环境类
    模拟车辆跟驰场景，智能体控制自车的加速度
    """
    
    def __init__(self, config=None, render_mode=None):
        """
        初始化环境
        
        Args:
            config: 配置字典，如果为None则使用默认配置
            render_mode: 渲染模式
        """
        super(ACCEnv, self).__init__()
        self.render_mode = render_mode
        
        # 导入配置
        if config is None:
            from config import (
                TARGET_SPEED, SAFETY_DISTANCE, MAX_ACCELERATION, 
                MAX_DECELERATION, DT, EPISODE_LENGTH, MAX_SPEED, MIN_SPEED
            )
        else:
            TARGET_SPEED = config.get('TARGET_SPEED', 25.0)
            SAFETY_DISTANCE = config.get('SAFETY_DISTANCE', 15.0)
            MAX_ACCELERATION = config.get('MAX_ACCELERATION', 2.0)
            MAX_DECELERATION = config.get('MAX_DECELERATION', -3.0)
            DT = config.get('DT', 0.1)
            EPISODE_LENGTH = config.get('EPISODE_LENGTH', 1000)
            MAX_SPEED = config.get('MAX_SPEED', 35.0)
            MIN_SPEED = config.get('MIN_SPEED', 0.0)
        
        # 保存配置
        self.target_speed = TARGET_SPEED
        self.safety_distance = SAFETY_DISTANCE
        self.max_acceleration = MAX_ACCELERATION
        self.max_deceleration = MAX_DECELERATION
        self.dt = DT
        self.episode_length = EPISODE_LENGTH
        self.max_speed = MAX_SPEED
        self.min_speed = MIN_SPEED
        
        # 定义动作空间和观测空间
        self.action_space = spaces.Box(
            low=np.array([self.max_deceleration], dtype=np.float32),
            high=np.array([self.max_acceleration], dtype=np.float32),
            dtype=np.float32
        )
        
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, -10.0, 0.0], dtype=np.float32),  # 自车速度、前车速度、相对距离、相对速度、目标速度
            high=np.array([self.max_speed, self.max_speed * 2, 200.0, 10.0, self.max_speed], dtype=np.float32),
            dtype=np.float32
        )
        
        # 初始化状态
        self.state = None
        self.current_step = 0
        self.episode_reward = 0
        
        # 历史记录用于可视化
        self.history = {
            'ego_speed': [],
            'lead_speed': [],
            'distance': [],
            'acceleration': [],
            'reward': []
        }
    
    def _get_observation(self):
        """
        获取当前观测值
        
        Returns:
            np.ndarray: 观测值数组
        """
        ego_speed, lead_speed, distance = self.state
        relative_speed = lead_speed - ego_speed
        
        return np.array([
            ego_speed,
            lead_speed,
            distance,
            relative_speed,
            self.target_speed
        ], dtype=np.float32)
    
    def _is_done(self):
        """
        判断是否达到终止条件
        
        Returns:
            bool: 是否终止
        """
        ego_speed, lead_speed, distance = self.state
        
        # 碰撞检测
        if distance < 0:
            return True
        
        # 距离过大（前车超出感知范围）
        if distance > 200:
            return True
        
        # 速度过低
        if ego_speed < 0.1 and self.current_step > 100:
            return True
        
        # 达到最大步数
        if self.current_step >= self.episode_length:
            return True
        
        return False
    
    def _lead_vehicle_dynamics(self, acceleration):
        """
        前车动力学模型
        
        Args:
            acceleration: 前车加速度
            
        Returns:
            float: 新的前车速度
        """
        ego_speed, lead_speed, distance = self.state
        
        # 更新前车速度
        new_lead_speed = lead_speed + acceleration * self.dt
        new_lead_speed = np.clip(new_lead_speed, 0.0, self.max_speed * 1.5)
        
        # 更新距离
        new_distance = distance + (lead_speed - ego_speed) * self.dt
        
        return new_lead_speed, new_distance
    
    def _ego_vehicle_dynamics(self, acceleration):
        """
        自车动力学模型
        
        Args:
            acceleration: 自车加速度指令
            
        Returns:
            float: 新的自车速度
        """
        ego_speed, _, _ = self.state
        
        # 更新自车速度
        new_ego_speed = ego_speed + acceleration * self.dt
        new_ego_speed = np.clip(new_ego_speed, self.min_speed, self.max_speed)
        
        return new_ego_speed
    
    def _generate_lead_vehicle_behavior(self):
        """
        生成前车行为（随机加减速）
        
        Returns:
            float: 前车加速度
        """
        # 每100步可能改变行为
        if self.current_step % 100 == 0:
            # 随机选择行为模式
            behavior = np.random.choice(['constant', 'accelerate', 'decelerate', 'random'], p=[0.4, 0.2, 0.2, 0.2])
            
            if behavior == 'constant':
                return 0.0
            elif behavior == 'accelerate':
                return np.random.uniform(0.5, 1.5)
            elif behavior == 'decelerate':
                return np.random.uniform(-2.0, -0.5)
            else:  # random
                return np.random.uniform(-1.0, 1.0)
        
        # 保持当前行为
        if hasattr(self, '_lead_acceleration'):
            return self._lead_acceleration
        else:
            return 0.0
    
    def reset(self, seed=None, options=None):
        """
        重置环境到初始状态
        
        Args:
            seed: 随机种子
            options: 额外选项
            
        Returns:
            np.ndarray: 初始观测值
            dict: 额外信息
        """
        super().reset(seed=seed)
        
        # 初始化状态
        self.ego_speed = np.random.uniform(self.target_speed * 0.8, self.target_speed * 1.2)
        self.lead_speed = np.random.uniform(self.target_speed * 0.8, self.target_speed * 1.2)
        self.distance = np.random.uniform(self.safety_distance * 1.5, self.safety_distance * 3)
        
        self.state = np.array([self.ego_speed, self.lead_speed, self.distance])
        self.current_step = 0
        self.episode_reward = 0
        
        # 重置历史记录
        self.history = {
            'ego_speed': [],
            'lead_speed': [],
            'distance': [],
            'acceleration': [],
            'reward': []
        }
        
        # 初始化前车行为
        self._lead_acceleration = self._generate_lead_vehicle_behavior()
        
        return self._get_observation(), {}
    
    def step(self, action):
        """
        执行一步环境交互
        
        Args:
            action: 动作（加速度指令）
            
        Returns:
            tuple: (观测值, 奖励, 是否终止, 是否截断, 信息)
        """
        # 确保动作在有效范围内
        acceleration = np.clip(action[0], self.max_deceleration, self.max_acceleration)
        
        # 获取当前状态
        ego_speed, lead_speed, distance = self.state
        
        # 更新自车速度
        new_ego_speed = self._ego_vehicle_dynamics(acceleration)
        
        # 生成前车行为
        lead_acceleration = self._generate_lead_vehicle_behavior()
        self._lead_acceleration = lead_acceleration
        
        # 更新前车状态
        new_lead_speed, new_distance = self._lead_vehicle_dynamics(lead_acceleration)
        
        # 更新相对距离
        new_distance = new_distance + (lead_speed - new_ego_speed) * self.dt
        
        # 更新状态
        self.state = np.array([new_ego_speed, new_lead_speed, new_distance])
        
        # 计算奖励
        reward = calculate_reward(
            new_ego_speed, new_lead_speed, new_distance, 
            acceleration, self.target_speed, self.safety_distance
        )
        
        # 更新历史记录
        self.history['ego_speed'].append(new_ego_speed)
        self.history['lead_speed'].append(new_lead_speed)
        self.history['distance'].append(new_distance)
        self.history['acceleration'].append(acceleration)
        self.history['reward'].append(reward)
        
        # 更新步数和总奖励
        self.current_step += 1
        self.episode_reward += reward
        
        # 获取观测值
        observation = self._get_observation()
        
        # 判断是否终止
        done = self._is_done()
        
        # 额外信息
        info = {
            'step': self.current_step,
            'episode_reward': self.episode_reward,
            'collision': new_distance < 0,
            'ego_speed': new_ego_speed,
            'lead_speed': new_lead_speed,
            'distance': new_distance
        }
        
        return observation, reward, done, False, info
    
    def render(self):
        """
        渲染环境（简单的文本输出）
        """
        if self.current_step % 100 == 0:
            ego_speed, lead_speed, distance = self.state
            print(f"Step: {self.current_step}")
            print(f"Ego Speed: {ego_speed:.2f} m/s")
            print(f"Lead Speed: {lead_speed:.2f} m/s")
            print(f"Distance: {distance:.2f} m")
            print(f"Episode Reward: {self.episode_reward:.2f}")
            print("-" * 50)
    
    def close(self):
        """
        关闭环境
        """
        pass

# 注册环境
register(
    id='ACC-v0',
    entry_point='acc_env.acc_env:ACCEnv',
    max_episode_steps=1000,
    kwargs={'render_mode': None}
)