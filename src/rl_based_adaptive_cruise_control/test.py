"""
测试脚本 - 使用训练好的模型进行测试
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

from acc_env import ACCEnv
from config import TEST_EPISODES, TEST_RENDER, MODEL_DIR

def parse_args():
    """
    解析命令行参数
    
    Returns:
        argparse.Namespace: 解析后的参数
    """
    parser = argparse.ArgumentParser(description='测试ACC强化学习模型')
    
    parser.add_argument('--model', type=str, default=None,
                      help='模型文件路径，如果不指定则使用最佳模型')
    
    parser.add_argument('--episodes', type=int, default=TEST_EPISODES,
                      help='测试回合数')
    
    parser.add_argument('--render', type=bool, default=TEST_RENDER,
                      help='是否渲染测试过程')
    
    parser.add_argument('--save-plots', type=bool, default=True,
                      help='是否保存图表')
    
    return parser.parse_args()

def load_model(model_path):
    """
    加载训练好的模型
    
    Args:
        model_path (str): 模型路径，如果为None则查找最佳模型
        
    Returns:
        PPO: 加载的模型
    """
    if model_path is None:
        # 查找最佳模型
        best_model_path = os.path.join(MODEL_DIR, "best_model.zip")
        if os.path.exists(best_model_path):
            model_path = best_model_path
        else:
            # 查找最新的模型
            existing_models = [f for f in os.listdir(MODEL_DIR) if f.endswith('.zip')]
            if existing_models:
                model_path = os.path.join(MODEL_DIR, max(existing_models, key=lambda x: os.path.getmtime(os.path.join(MODEL_DIR, x))))
            else:
                raise FileNotFoundError("没有找到训练好的模型")
    
    print(f"📂 加载模型: {model_path}")
    model = PPO.load(model_path)
    
    return model

def test_model(model, env, num_episodes=5, render=False, save_plots=True):
    """
    测试模型性能
    
    Args:
        model (PPO): 训练好的模型
        env (ACCEnv): 测试环境
        num_episodes (int): 测试回合数
        render (bool): 是否渲染测试过程
        save_plots (bool): 是否保存图表
        
    Returns:
        dict: 测试结果统计
    """
    results = {
        'total_rewards': [],
        'collisions': 0,
        'success_episodes': 0,
        'avg_speed_error': [],
        'avg_distance_error': [],
        'histories': []
    }
    
    for episode in range(num_episodes):
        print(f"\n🚗 测试回合 {episode + 1}/{num_episodes}")
        print("-" * 50)
        
        obs, _ = env.reset()
        episode_reward = 0
        done = False
        collision = False
        
        step = 0
        
        while not done:
            # 使用模型预测动作
            action, _ = model.predict(obs, deterministic=True)
            
            # 执行动作
            obs, reward, done, _, info = env.step(action)
            episode_reward += reward
            
            step += 1
            
            # 检查碰撞
            if info.get('collision', False):
                collision = True
            
            # 渲染
            if render and step % 10 == 0:
                env.render()
        
        # 保存历史记录
        results['histories'].append(env.history.copy())
        
        # 计算统计信息
        ego_speeds = np.array(env.history['ego_speed'])
        lead_speeds = np.array(env.history['lead_speed'])
        distances = np.array(env.history['distance'])
        
        speed_error = np.abs(ego_speeds - env.target_speed)
        desired_distances = env.safety_distance + ego_speeds * 1.5
        distance_error = np.abs(distances - desired_distances)
        
        results['total_rewards'].append(episode_reward)
        results['avg_speed_error'].append(np.mean(speed_error))
        results['avg_distance_error'].append(np.mean(distance_error))
        
        if collision:
            results['collisions'] += 1
            print(f"💥 碰撞发生！回合奖励: {episode_reward:.2f}")
        else:
            results['success_episodes'] += 1
            print(f"✅ 成功完成！回合奖励: {episode_reward:.2f}")
        
        print(f"⏱️  持续时间: {step * env.dt:.1f} 秒")
        print(f"📊 平均速度误差: {np.mean(speed_error):.2f} m/s")
        print(f"📏 平均距离误差: {np.mean(distance_error):.2f} m")
    
    # 计算总体统计
    results['avg_reward'] = np.mean(results['total_rewards'])
    results['std_reward'] = np.std(results['total_rewards'])
    results['success_rate'] = results['success_episodes'] / num_episodes
    results['avg_speed_error_overall'] = np.mean(results['avg_speed_error'])
    results['avg_distance_error_overall'] = np.mean(results['avg_distance_error'])
    
    return results

def plot_test_results(results, save_path='test_results'):
    """
    绘制测试结果图表
    
    Args:
        results (dict): 测试结果
        save_path (str): 图表保存路径
    """
    os.makedirs(save_path, exist_ok=True)
    
    # 绘制奖励分布
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(results['total_rewards'])), results['total_rewards'])
    plt.xlabel('回合')
    plt.ylabel('总奖励')
    plt.title('各回合奖励分布')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_path, 'rewards.png'))
    plt.close()
    
    # 绘制速度跟踪图（选择第一个成功的回合）
    success_idx = None
    for i, history in enumerate(results['histories']):
        if len(history['ego_speed']) > 0:
            success_idx = i
            break
    
    if success_idx is not None:
        history = results['histories'][success_idx]
        time = np.arange(len(history['ego_speed'])) * 0.1
        
        plt.figure(figsize=(12, 8))
        
        # 速度跟踪
        plt.subplot(3, 1, 1)
        plt.plot(time, history['ego_speed'], label='自车速度')
        plt.plot(time, history['lead_speed'], label='前车速度')
        plt.axhline(y=history.get('target_speed', 25), color='r', linestyle='--', label='目标速度')
        plt.xlabel('时间 (s)')
        plt.ylabel('速度 (m/s)')
        plt.title('速度跟踪')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 距离变化
        plt.subplot(3, 1, 2)
        plt.plot(time, history['distance'], label='实际距离')
        safety_distance = np.array(history['ego_speed']) * 1.5 + 15  # 15m + 1.5s时间间隔
        plt.plot(time[:len(safety_distance)], safety_distance, 'r--', label='期望安全距离')
        plt.xlabel('时间 (s)')
        plt.ylabel('距离 (m)')
        plt.title('车间距离')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 加速度和奖励
        plt.subplot(3, 1, 3)
        plt.plot(time, history['acceleration'], 'g-', label='加速度')
        plt.xlabel('时间 (s)')
        plt.ylabel('加速度 (m/s²)')
        plt.title('加速度变化')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'performance.png'))
        plt.close()
    
    # 绘制统计信息
    plt.figure(figsize=(10, 6))
    metrics = ['成功率', '平均速度误差', '平均距离误差']
    values = [
        results['success_rate'] * 100,
        results['avg_speed_error_overall'],
        results['avg_distance_error_overall']
    ]
    
    plt.bar(metrics, values)
    plt.ylabel('数值')
    plt.title('测试性能统计')
    plt.grid(True, alpha=0.3, axis='y')
    
    # 在柱状图上显示数值
    for i, v in enumerate(values):
        if metrics[i] == '成功率':
            plt.text(i, v + 1, f'{v:.1f}%', ha='center')
        else:
            plt.text(i, v + 0.1, f'{v:.2f}', ha='center')
    
    plt.savefig(os.path.join(save_path, 'statistics.png'))
    plt.close()

def print_test_summary(results):
    """
    打印测试结果摘要
    
    Args:
        results (dict): 测试结果
    """
    print("\n" + "=" * 50)
    print("📊 测试结果摘要")
    print("=" * 50)
    
    print(f"平均奖励: {results['avg_reward']:.2f} ± {results['std_reward']:.2f}")
    print(f"成功率: {results['success_rate']:.1%} ({results['success_episodes']}/{len(results['total_rewards'])})")
    print(f"碰撞次数: {results['collisions']}")
    print(f"平均速度误差: {results['avg_speed_error_overall']:.2f} m/s")
    print(f"平均距离误差: {results['avg_distance_error_overall']:.2f} m")

def main():
    """
    主函数
    """
    args = parse_args()
    
    try:
        # 加载模型
        model = load_model(args.model)
        
        # 创建环境
        env = ACCEnv(render_mode='human' if args.render else None)
        
        # 测试模型
        results = test_model(
            model=model,
            env=env,
            num_episodes=args.episodes,
            render=args.render,
            save_plots=args.save_plots
        )
        
        # 打印结果摘要
        print_test_summary(results)
        
        # 绘制结果图表
        if args.save_plots:
            plot_test_results(results)
            print(f"\n📈 图表已保存到 'test_results/' 目录")
        
    except Exception as e:
        print(f"❌ 测试过程中发生错误: {str(e)}")
    finally:
        if 'env' in locals():
            env.close()

if __name__ == "__main__":
    main()