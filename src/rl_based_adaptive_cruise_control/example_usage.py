"""
示例使用脚本 - 展示如何使用ACC环境和训练模型
"""

import numpy as np
from stable_baselines3 import PPO

from acc_env import ACCEnv
from config import TARGET_SPEED, SAFETY_DISTANCE

def random_policy_demo():
    """
    演示使用随机策略控制ACC
    """
    print("🎲 随机策略演示")
    print("-" * 50)
    
    env = ACCEnv()
    obs, _ = env.reset()
    done = False
    total_reward = 0
    steps = 0
    
    while not done and steps < 500:
        # 随机选择加速度
        action = env.action_space.sample()
        
        # 执行动作
        obs, reward, done, _, info = env.step(action)
        total_reward += reward
        steps += 1
        
        # 每50步打印一次状态
        if steps % 50 == 0:
            print(f"步骤 {steps}:")
            print(f"  自车速度: {info['ego_speed']:.2f} m/s")
            print(f"  前车速度: {info['lead_speed']:.2f} m/s")
            print(f"  车间距离: {info['distance']:.2f} m")
            print(f"  当前奖励: {reward:.2f}")
            print(f"  总奖励: {total_reward:.2f}")
            print("-" * 30)
    
    print(f"\n演示结束！")
    print(f"总步数: {steps}")
    print(f"总奖励: {total_reward:.2f}")
    print(f"平均奖励: {total_reward/steps:.2f}")
    
    env.close()

def constant_speed_demo():
    """
    演示使用恒速策略控制ACC
    """
    print("\n🚗 恒速策略演示")
    print("-" * 50)
    
    env = ACCEnv()
    obs, _ = env.reset()
    done = False
    total_reward = 0
    steps = 0
    
    while not done and steps < 500:
        ego_speed, lead_speed, distance, relative_speed, target_speed = obs
        
        # 简单的恒速控制策略
        if ego_speed < target_speed:
            acceleration = 1.0  # 加速
        else:
            acceleration = -0.5  # 减速
        
        # 安全距离检查
        desired_distance = SAFETY_DISTANCE + ego_speed * 1.5
        if distance < desired_distance:
            acceleration = min(acceleration, -1.0)  # 更激进地减速
        
        action = np.array([acceleration])
        
        # 执行动作
        obs, reward, done, _, info = env.step(action)
        total_reward += reward
        steps += 1
        
        # 每50步打印一次状态
        if steps % 50 == 0:
            print(f"步骤 {steps}:")
            print(f"  自车速度: {info['ego_speed']:.2f} m/s")
            print(f"  前车速度: {info['lead_speed']:.2f} m/s")
            print(f"  车间距离: {info['distance']:.2f} m")
            print(f"  加速度: {acceleration:.2f} m/s²")
            print(f"  当前奖励: {reward:.2f}")
            print(f"  总奖励: {total_reward:.2f}")
            print("-" * 30)
    
    print(f"\n演示结束！")
    print(f"总步数: {steps}")
    print(f"总奖励: {total_reward:.2f}")
    print(f"平均奖励: {total_reward/steps:.2f}")
    
    env.close()

def trained_model_demo(model_path=None):
    """
    演示使用训练好的模型控制ACC
    
    Args:
        model_path (str): 模型路径
    """
    try:
        print("\n🤖 训练模型演示")
        print("-" * 50)
        
        # 尝试加载模型
        if model_path is None:
            try:
                model = PPO.load("models/best_model.zip")
                print("✅ 成功加载训练好的模型")
            except:
                print("⚠️  没有找到训练好的模型，使用PPO默认参数创建新模型")
                from stable_baselines3 import PPO
                env = ACCEnv()
                model = PPO("MlpPolicy", env, verbose=0)
        else:
            model = PPO.load(model_path)
            print(f"✅ 成功加载模型: {model_path}")
        
        # 创建环境
        env = ACCEnv()
        obs, _ = env.reset()
        done = False
        total_reward = 0
        steps = 0
        
        while not done and steps < 500:
            # 使用模型预测动作
            action, _ = model.predict(obs, deterministic=True)
            
            # 执行动作
            obs, reward, done, _, info = env.step(action)
            total_reward += reward
            steps += 1
            
            # 每50步打印一次状态
            if steps % 50 == 0:
                print(f"步骤 {steps}:")
                print(f"  自车速度: {info['ego_speed']:.2f} m/s")
                print(f"  前车速度: {info['lead_speed']:.2f} m/s")
                print(f"  车间距离: {info['distance']:.2f} m")
                print(f"  加速度: {action[0]:.2f} m/s²")
                print(f"  当前奖励: {reward:.2f}")
                print(f"  总奖励: {total_reward:.2f}")
                print("-" * 30)
        
        print(f"\n演示结束！")
        print(f"总步数: {steps}")
        print(f"总奖励: {total_reward:.2f}")
        print(f"平均奖励: {total_reward/steps:.2f}")
        
        # 保存历史记录用于分析
        return env.history
        
    except Exception as e:
        print(f"❌ 模型演示失败: {str(e)}")
        return None
    finally:
        if 'env' in locals():
            env.close()

def compare_policies():
    """
    比较不同策略的性能
    """
    print("📊 策略比较")
    print("=" * 60)
    
    # 运行多次取平均
    num_runs = 3
    
    results = {
        'random': {'rewards': [], 'steps': []},
        'constant': {'rewards': [], 'steps': []},
        'trained': {'rewards': [], 'steps': []}
    }
    
    for run in range(num_runs):
        print(f"\n--- 运行 {run + 1}/{num_runs} ---")
        
        # 随机策略
        env = ACCEnv()
        obs, _ = env.reset()
        done = False
        random_reward = 0
        random_steps = 0
        
        while not done and random_steps < 300:
            action = env.action_space.sample()
            obs, reward, done, _, _ = env.step(action)
            random_reward += reward
            random_steps += 1
        
        results['random']['rewards'].append(random_reward)
        results['random']['steps'].append(random_steps)
        env.close()
        
        # 恒速策略
        env = ACCEnv()
        obs, _ = env.reset()
        done = False
        constant_reward = 0
        constant_steps = 0
        
        while not done and constant_steps < 300:
            ego_speed, lead_speed, distance, relative_speed, target_speed = obs
            
            if ego_speed < target_speed:
                acceleration = 1.0
            else:
                acceleration = -0.5
            
            desired_distance = SAFETY_DISTANCE + ego_speed * 1.5
            if distance < desired_distance:
                acceleration = min(acceleration, -1.0)
            
            action = np.array([acceleration])
            obs, reward, done, _, _ = env.step(action)
            constant_reward += reward
            constant_steps += 1
        
        results['constant']['rewards'].append(constant_reward)
        results['constant']['steps'].append(constant_steps)
        env.close()
        
        # 训练模型（如果可用）
        try:
            model = PPO.load("models/best_model.zip")
            env = ACCEnv()
            obs, _ = env.reset()
            done = False
            trained_reward = 0
            trained_steps = 0
            
            while not done and trained_steps < 300:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, _, _ = env.step(action)
                trained_reward += reward
                trained_steps += 1
            
            results['trained']['rewards'].append(trained_reward)
            results['trained']['steps'].append(trained_steps)
            env.close()
        except:
            print("⚠️  跳过训练模型比较（模型不可用）")
    
    # 打印结果
    print("\n📋 比较结果:")
    print("-" * 50)
    
    for policy in results:
        if results[policy]['rewards']:
            avg_reward = np.mean(results[policy]['rewards'])
            avg_steps = np.mean(results[policy]['steps'])
            avg_reward_per_step = avg_reward / avg_steps
            
            print(f"{policy.capitalize()} 策略:")
            print(f"  平均奖励: {avg_reward:.2f}")
            print(f"  平均步数: {avg_steps:.1f}")
            print(f"  每步平均奖励: {avg_reward_per_step:.3f}")
            print("-" * 30)

def main():
    """
    主函数
    """
    print("🚗 自适应巡航控制 - 示例使用")
    print("=" * 60)
    
    # 运行各种演示
    random_policy_demo()
    constant_speed_demo()
    trained_model_demo()
    compare_policies()
    
    print("\n🎉 所有演示完成！")
    print("提示:")
    print("  - 运行 'python train.py' 来训练模型")
    print("  - 运行 'python test.py' 来测试训练好的模型")
    print("  - 运行 'python visualize.py' 来可视化结果")

if __name__ == "__main__":
    main()