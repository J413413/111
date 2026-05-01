"""
奖励函数定义
"""

import numpy as np

def calculate_reward(ego_speed, lead_speed, distance, acceleration, target_speed, safety_distance):
    """
    计算当前状态的奖励值
    
    Args:
        ego_speed (float): 自车速度 (m/s)
        lead_speed (float): 前车速度 (m/s)
        distance (float): 两车之间的距离 (m)
        acceleration (float): 自车加速度 (m/s²)
        target_speed (float): 目标速度 (m/s)
        safety_distance (float): 安全距离 (m)
        
    Returns:
        float: 奖励值
    """
    reward = 0.0
    
    # 1. 速度跟踪奖励
    speed_error = abs(ego_speed - target_speed)
    speed_reward = -0.1 * speed_error
    
    # 当速度接近目标速度时给予额外奖励
    if speed_error < 1.0:
        speed_reward += 0.5
    
    # 2. 安全距离奖励
    desired_distance = safety_distance + ego_speed * 1.5  # 时间间隔1.5秒
    distance_error = abs(distance - desired_distance)
    
    if distance < 0:
        # 碰撞惩罚
        distance_reward = -100.0
    elif distance < safety_distance:
        # 距离过近惩罚
        distance_reward = -5.0 - 0.5 * (safety_distance - distance)
    elif distance > 150:
        # 距离过远惩罚
        distance_reward = -2.0
    else:
        # 保持合适距离奖励
        distance_reward = 1.0 - 0.01 * distance_error
    
    # 3. 乘坐舒适度奖励（加速度平滑性）
    acceleration_penalty = -0.5 * abs(acceleration)
    
    # 4. 相对速度奖励（避免急加速/减速）
    relative_speed = lead_speed - ego_speed
    relative_speed_reward = -0.1 * abs(relative_speed)
    
    # 5. 效率奖励（保持较高速度）
    efficiency_reward = 0.05 * ego_speed
    
    # 6. 碰撞避免奖励
    if distance > safety_distance + ego_speed * 0.5:
        collision_avoidance_reward = 0.5
    else:
        collision_avoidance_reward = 0.0
    
    # 综合奖励
    reward = (
        speed_reward +
        distance_reward +
        acceleration_penalty +
        relative_speed_reward +
        efficiency_reward +
        collision_avoidance_reward
    )
    
    # 额外的特殊情况惩罚
    if ego_speed < 0:
        reward -= 10.0  # 速度为负惩罚
    
    if acceleration > 3.0 or acceleration < -4.0:
        reward -= 2.0  # 过激加速度惩罚
    
    return reward

def calculate_reward_v2(ego_speed, lead_speed, distance, acceleration, target_speed, safety_distance, prev_acceleration=None):
    """
    改进版奖励函数，增加了加速度变化率的考虑
    
    Args:
        ego_speed (float): 自车速度 (m/s)
        lead_speed (float): 前车速度 (m/s)
        distance (float): 两车之间的距离 (m)
        acceleration (float): 自车加速度 (m/s²)
        target_speed (float): 目标速度 (m/s)
        safety_distance (float): 安全距离 (m)
        prev_acceleration (float): 上一步的加速度 (m/s²)
        
    Returns:
        float: 奖励值
    """
    reward = calculate_reward(ego_speed, lead_speed, distance, acceleration, target_speed, safety_distance)
    
    # 增加加速度变化率惩罚（提高乘坐舒适度）
    if prev_acceleration is not None:
        jerk_penalty = -0.3 * abs(acceleration - prev_acceleration)
        reward += jerk_penalty
    
    return reward

def calculate_reward_v3(state, action, target_speed, safety_distance, prev_action=None):
    """
    基于状态和动作的奖励函数，更加灵活
    
    Args:
        state (np.ndarray): 当前状态 [ego_speed, lead_speed, distance]
        action (np.ndarray): 当前动作 [acceleration]
        target_speed (float): 目标速度
        safety_distance (float): 安全距离
        prev_action (np.ndarray): 上一个动作
        
    Returns:
        float: 奖励值
    """
    ego_speed, lead_speed, distance = state
    acceleration = action[0]
    
    # 基础奖励
    reward = calculate_reward(ego_speed, lead_speed, distance, acceleration, target_speed, safety_distance)
    
    # 增加更多的策略性奖励
    
    # 1. 前车减速时的及时响应奖励
    if lead_speed < ego_speed and distance < safety_distance + ego_speed:
        if acceleration < 0:
            reward += 1.0
    
    # 2. 前车加速时的跟随奖励
    if lead_speed > ego_speed and distance > safety_distance:
        if acceleration > 0:
            reward += 0.5
    
    # 3. 交通流效率奖励
    traffic_efficiency = ego_speed / (lead_speed + 1e-6)
    if 0.8 < traffic_efficiency < 1.2:
        reward += 0.3
    
    return reward