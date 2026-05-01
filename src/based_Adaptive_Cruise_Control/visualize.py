"""
可视化脚本 - 用于可视化训练结果和模型性能
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from stable_baselines3 import PPO

from acc_env import ACCEnv
from config import MODEL_DIR

def parse_args():
    """
    解析命令行参数
    
    Returns:
        argparse.Namespace: 解析后的参数
    """
    parser = argparse.ArgumentParser(description='可视化ACC模型性能')
    
    parser.add_argument('--model', type=str, default=None,
                      help='模型文件路径，如果不指定则使用最佳模型')
    
    parser.add_argument('--episodes', type=int, default=3,
                      help='可视化的回合数')
    
    parser.add_argument('--save', type=bool, default=True,
                      help='是否保存可视化结果')
    
    parser.add_argument('--format', type=str, default='png',
                      choices=['png', 'pdf', 'svg'],
                      help='保存格式')
    
    return parser.parse_args()

def load_model(model_path):
    """
    加载训练好的模型
    
    Args:
        model_path (str): 模型路径
        
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

def collect_episode_data(model, env, num_episodes=1):
    """
    收集回合数据用于可视化
    
    Args:
        model (PPO): 训练好的模型
        env (ACCEnv): 环境
        num_episodes (int): 收集的回合数
        
    Returns:
        list: 回合历史记录列表
    """
    histories = []
    
    for episode in range(num_episodes):
        print(f"收集回合 {episode + 1}/{num_episodes} 的数据...")
        
        obs, _ = env.reset()
        done = False
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, _, _ = env.step(action)
        
        histories.append(env.history.copy())
    
    return histories

def plot_time_series(history, save_path=None, format='png'):
    """
    绘制时间序列图表
    
    Args:
        history (dict): 历史记录
        save_path (str): 保存路径
        format (str): 保存格式
    """
    time = np.arange(len(history['ego_speed'])) * 0.1
    
    plt.figure(figsize=(14, 10))
    
    # 速度跟踪
    plt.subplot(4, 1, 1)
    plt.plot(time, history['ego_speed'], 'b-', linewidth=2, label='自车速度')
    plt.plot(time, history['lead_speed'], 'g-', linewidth=2, label='前车速度')
    plt.axhline(y=env.target_speed, color='r', linestyle='--', linewidth=2, label='目标速度')
    plt.fill_between(time, history['ego_speed'], alpha=0.1, color='blue')
    plt.xlabel('时间 (s)')
    plt.ylabel('速度 (m/s)')
    plt.title('速度跟踪性能')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.minorticks_on()
    plt.grid(which='minor', alpha=0.1)
    
    # 车间距离
    plt.subplot(4, 1, 2)
    plt.plot(time, history['distance'], 'b-', linewidth=2, label='实际距离')
    desired_distance = env.safety_distance + np.array(history['ego_speed']) * 1.5
    plt.plot(time, desired_distance, 'r--', linewidth=2, label='期望安全距离')
    plt.axhline(y=env.safety_distance, color='orange', linestyle='-.', linewidth=1.5, label='最小安全距离')
    plt.fill_between(time, history['distance'], alpha=0.1, color='blue')
    plt.xlabel('时间 (s)')
    plt.ylabel('距离 (m)')
    plt.title('车间距离控制')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.minorticks_on()
    plt.grid(which='minor', alpha=0.1)
    
    # 加速度
    plt.subplot(4, 1, 3)
    plt.plot(time, history['acceleration'], 'g-', linewidth=2, label='加速度')
    plt.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    plt.fill_between(time, history['acceleration'], alpha=0.2, color='green')
    plt.xlabel('时间 (s)')
    plt.ylabel('加速度 (m/s²)')
    plt.title('加速度控制')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.minorticks_on()
    plt.grid(which='minor', alpha=0.1)
    
    # 奖励
    plt.subplot(4, 1, 4)
    plt.plot(time, history['reward'], 'm-', linewidth=1.5, label='即时奖励')
    cumulative_reward = np.cumsum(history['reward'])
    plt.plot(time, cumulative_reward, 'c-', linewidth=2, label='累积奖励')
    plt.fill_between(time, history['reward'], alpha=0.2, color='magenta')
    plt.xlabel('时间 (s)')
    plt.ylabel('奖励值')
    plt.title('奖励信号')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.minorticks_on()
    plt.grid(which='minor', alpha=0.1)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(f"{save_path}_time_series.{format}", dpi=300, bbox_inches='tight')
        print(f"📈 时间序列图已保存到: {save_path}_time_series.{format}")
    
    plt.show()

def plot_phase_diagrams(history, save_path=None, format='png'):
    """
    绘制相图
    
    Args:
        history (dict): 历史记录
        save_path (str): 保存路径
        format (str): 保存格式
    """
    plt.figure(figsize=(15, 5))
    
    # 速度-距离相图
    plt.subplot(1, 3, 1)
    plt.scatter(history['distance'], history['ego_speed'], c=np.arange(len(history['ego_speed'])), 
                cmap='viridis', alpha=0.6, s=10)
    plt.colorbar(label='时间步')
    plt.xlabel('车间距离 (m)')
    plt.ylabel('自车速度 (m/s)')
    plt.title('速度-距离相图')
    plt.grid(True, alpha=0.3)
    
    # 速度-加速度相图
    plt.subplot(1, 3, 2)
    plt.scatter(history['ego_speed'], history['acceleration'], 
                c=np.arange(len(history['ego_speed'])), cmap='plasma', alpha=0.6, s=10)
    plt.colorbar(label='时间步')
    plt.xlabel('自车速度 (m/s)')
    plt.ylabel('加速度 (m/s²)')
    plt.title('速度-加速度相图')
    plt.grid(True, alpha=0.3)
    
    # 相对速度-距离相图
    plt.subplot(1, 3, 3)
    relative_speed = np.array(history['lead_speed']) - np.array(history['ego_speed'])
    plt.scatter(history['distance'], relative_speed, 
                c=np.arange(len(history['ego_speed'])), cmap='inferno', alpha=0.6, s=10)
    plt.colorbar(label='时间步')
    plt.xlabel('车间距离 (m)')
    plt.ylabel('相对速度 (m/s)')
    plt.title('相对速度-距离相图')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(f"{save_path}_phase_diagrams.{format}", dpi=300, bbox_inches='tight')
        print(f"📊 相图已保存到: {save_path}_phase_diagrams.{format}")
    
    plt.show()

def create_animation(history, save_path=None):
    """
    创建动画
    
    Args:
        history (dict): 历史记录
        save_path (str): 保存路径
    """
    try:
        from matplotlib.animation import FuncAnimation
    except ImportError:
        print("⚠️ matplotlib.animation 导入失败，跳过动画创建")
        return
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    fig.subplots_adjust(hspace=0.3)
    
    time = np.arange(len(history['ego_speed'])) * 0.1
    max_time = time[-1]
    
    # 初始化图形
    line1, = ax1.plot([], [], 'b-', linewidth=2, label='自车速度')
    line2, = ax1.plot([], [], 'g-', linewidth=2, label='前车速度')
    line3, = ax1.plot([], [], 'r--', linewidth=2, label='目标速度')
    line4, = ax2.plot([], [], 'b-', linewidth=2, label='实际距离')
    line5, = ax2.plot([], [], 'r--', linewidth=2, label='期望距离')
    
    def init():
        ax1.set_xlim(0, max_time)
        ax1.set_ylim(0, max(max(history['ego_speed']), max(history['lead_speed'])) * 1.1)
        ax1.set_xlabel('时间 (s)')
        ax1.set_ylabel('速度 (m/s)')
        ax1.set_title('速度跟踪')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.set_xlim(0, max_time)
        ax2.set_ylim(0, max(history['distance']) * 1.1)
        ax2.set_xlabel('时间 (s)')
        ax2.set_ylabel('距离 (m)')
        ax2.set_title('车间距离')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        return line1, line2, line3, line4, line5
    
    def update(frame):
        current_time = time[:frame+1]
        
        line1.set_data(current_time, history['ego_speed'][:frame+1])
        line2.set_data(current_time, history['lead_speed'][:frame+1])
        line3.set_data([0, max_time], [env.target_speed, env.target_speed])
        
        desired_distance = env.safety_distance + np.array(history['ego_speed'][:frame+1]) * 1.5
        line4.set_data(current_time, history['distance'][:frame+1])
        line5.set_data(current_time, desired_distance)
        
        return line1, line2, line3, line4, line5
    
    ani = FuncAnimation(fig, update, frames=len(time), init_func=init,
                       interval=50, blit=True)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        ani.save(f"{save_path}_animation.gif", writer='pillow', fps=20)
        print(f"🎬 动画已保存到: {save_path}_animation.gif")
    
    plt.show()

def plot_distributions(history, save_path=None, format='png'):
    """
    绘制分布图表
    
    Args:
        history (dict): 历史记录
        save_path (str): 保存路径
        format (str): 保存格式
    """
    plt.figure(figsize=(15, 10))
    
    # 速度分布
    plt.subplot(2, 3, 1)
    plt.hist(history['ego_speed'], bins=30, alpha=0.7, color='blue', edgecolor='black')
    plt.axvline(x=env.target_speed, color='red', linestyle='--', linewidth=2, label='目标速度')
    plt.xlabel('速度 (m/s)')
    plt.ylabel('频次')
    plt.title('自车速度分布')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 加速度分布
    plt.subplot(2, 3, 2)
    plt.hist(history['acceleration'], bins=30, alpha=0.7, color='green', edgecolor='black')
    plt.axvline(x=0, color='red', linestyle='--', linewidth=1)
    plt.xlabel('加速度 (m/s²)')
    plt.ylabel('频次')
    plt.title('加速度分布')
    plt.grid(True, alpha=0.3)
    
    # 距离分布
    plt.subplot(2, 3, 3)
    plt.hist(history['distance'], bins=30, alpha=0.7, color='purple', edgecolor='black')
    plt.axvline(x=env.safety_distance, color='orange', linestyle='--', linewidth=2, label='安全距离')
    plt.xlabel('距离 (m)')
    plt.ylabel('频次')
    plt.title('车间距离分布')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 奖励分布
    plt.subplot(2, 3, 4)
    plt.hist(history['reward'], bins=30, alpha=0.7, color='gold', edgecolor='black')
    plt.xlabel('奖励值')
    plt.ylabel('频次')
    plt.title('奖励分布')
    plt.grid(True, alpha=0.3)
    
    # 速度误差分布
    plt.subplot(2, 3, 5)
    speed_error = np.abs(np.array(history['ego_speed']) - env.target_speed)
    plt.hist(speed_error, bins=30, alpha=0.7, color='red', edgecolor='black')
    plt.xlabel('速度误差 (m/s)')
    plt.ylabel('频次')
    plt.title('速度误差分布')
    plt.grid(True, alpha=0.3)
    
    # 距离误差分布
    plt.subplot(2, 3, 6)
    desired_distance = env.safety_distance + np.array(history['ego_speed']) * 1.5
    distance_error = np.abs(np.array(history['distance']) - desired_distance)
    plt.hist(distance_error, bins=30, alpha=0.7, color='brown', edgecolor='black')
    plt.xlabel('距离误差 (m)')
    plt.ylabel('频次')
    plt.title('距离误差分布')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(f"{save_path}_distributions.{format}", dpi=300, bbox_inches='tight')
        print(f"📊 分布图已保存到: {save_path}_distributions.{format}")
    
    plt.show()

def main():
    """
    主函数
    """
    global env
    
    args = parse_args()
    
    try:
        # 创建保存目录
        save_dir = 'visualization_results'
        os.makedirs(save_dir, exist_ok=True)
        
        # 加载模型
        model = load_model(args.model)
        
        # 创建环境
        env = ACCEnv(render_mode=None)
        
        # 收集数据
        histories = collect_episode_data(model, env, args.episodes)
        
        for i, history in enumerate(histories):
            print(f"\n📊 可视化回合 {i + 1} 的结果")
            
            save_path = os.path.join(save_dir, f'episode_{i+1}')
            
            # 绘制时间序列图
            plot_time_series(history, save_path=save_path, format=args.format)
            
            # 绘制相图
            plot_phase_diagrams(history, save_path=save_path, format=args.format)
            
            # 绘制分布图
            plot_distributions(history, save_path=save_path, format=args.format)
            
            # 创建动画（第一个回合）
            if i == 0:
                create_animation(history, save_path=save_path)
        
        print(f"\n🎉 所有可视化结果已保存到 '{save_dir}' 目录")
        
    except Exception as e:
        print(f"❌ 可视化过程中发生错误: {str(e)}")
    finally:
        if 'env' in locals():
            env.close()

if __name__ == "__main__":
    main()