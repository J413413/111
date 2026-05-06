"""
训练脚本 - 使用PPO算法训练ACC智能体
"""

import os
import time
import numpy as np
import torch
from tqdm import tqdm
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from acc_env import ACCEnv
from config import (
    TRAINING_TIMESTEPS, SAVE_FREQUENCY, LOG_FREQUENCY,
    MODEL_DIR, LOG_DIR, BATCH_SIZE, GAMMA, LEARNING_RATE,
    N_EPOCHS, CLIP_RANGE, TARGET_SPEED, SAFETY_DISTANCE
)

def make_env():
    """
    创建环境函数
    
    Returns:
        gym.Env: 环境实例
    """
    env = ACCEnv()
    return env

def train_ppo_model():
    """
    训练PPO模型
    """
    print("=" * 60)
    print("🚗 开始训练自适应巡航控制智能体")
    print("=" * 60)
    
    # 创建目录
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    
    # 创建环境
    env = make_vec_env(make_env, n_envs=1)
    
    # 创建评估环境
    eval_env = ACCEnv()    
    # 定义模型参数
    model_kwargs = {
        'learning_rate': LEARNING_RATE,
        'n_steps': 2048,
        'batch_size': BATCH_SIZE,
        'n_epochs': N_EPOCHS,
        'gamma': GAMMA,
        'gae_lambda': 0.95,
        'clip_range': CLIP_RANGE,
        'clip_range_vf': None,
        'normalize_advantage': True,
        'ent_coef': 0.0,
        'vf_coef': 0.5,
        'max_grad_norm': 0.5,
        'use_sde': False,
        'sde_sample_freq': -1,
        'target_kl': None,
        'device': 'auto',
        'policy_kwargs': {
            'net_arch': [64, 64],
            'activation_fn': torch.nn.ReLU,
            'ortho_init': True,
            'log_std_init': 0.0,
            'full_std': True,
            'squash_output': False,
            'share_features_extractor': False
        }
    }
    
    # 创建模型
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=LOG_DIR,
        **model_kwargs
    )
    
    # 加载已有模型（如果存在）
    existing_models = [f for f in os.listdir(MODEL_DIR) if f.endswith('.zip')]
    if existing_models:
        latest_model = max(existing_models, key=lambda x: os.path.getmtime(os.path.join(MODEL_DIR, x)))
        print(f"🔄 加载已有模型: {latest_model}")
        model = PPO.load(os.path.join(MODEL_DIR, latest_model), env=env)
    
    # 定义回调函数
    checkpoint_callback = CheckpointCallback(
        save_freq=SAVE_FREQUENCY,
        save_path=MODEL_DIR,
        name_prefix="acc_model",
        save_replay_buffer=False,
        save_vecnormalize=False
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=MODEL_DIR,
        log_path=LOG_DIR,
        eval_freq=SAVE_FREQUENCY // 2,
        deterministic=True,
        render=False,
        n_eval_episodes=5
    )
    
    # 开始训练
    start_time = time.time()
    
    try:
        model.learn(
            total_timesteps=TRAINING_TIMESTEPS,
            callback=[checkpoint_callback, eval_callback],
            log_interval=LOG_FREQUENCY // 10,
            tb_log_name="ppo_acc_training",
            reset_num_timesteps=False
        )
        
    except KeyboardInterrupt:
        print("\n⏹️ 训练被用户中断")
    
    # 保存最终模型
    final_model_path = os.path.join(MODEL_DIR, "final_model.zip")
    model.save(final_model_path)
    print(f"\n💾 最终模型已保存到: {final_model_path}")
    
    # 计算训练时间
    training_time = time.time() - start_time
    print(f"\n⏱️ 训练总时间: {training_time:.2f} 秒")
    
    # 关闭环境
    env.close()
    eval_env.close()
    
    return model

def test_trained_model(model_path=None):
    """
    测试训练好的模型
    
    Args:
        model_path (str): 模型路径，如果为None则使用最佳模型
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
                print("❌ 没有找到训练好的模型")
                return
    
    print(f"\n🧪 测试模型: {model_path}")
    
    # 加载模型
    model = PPO.load(model_path)
    
    # 创建测试环境
    env = ACCEnv()
    
    total_rewards = []
    success_episodes = 0
    collision_episodes = 0
    
    for episode in range(5):
        obs, _ = env.reset()
        episode_reward = 0
        done = False
        collision = False
        
        print(f"\n--- 测试回合 {episode + 1} ---")
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, info = env.step(action)
            episode_reward += reward
            
            if info.get('collision', False):
                collision = True
        
        total_rewards.append(episode_reward)
        
        if collision:
            collision_episodes += 1
            print(f"💥 碰撞发生！回合奖励: {episode_reward:.2f}")
        else:
            success_episodes += 1
            print(f"✅ 成功完成！回合奖励: {episode_reward:.2f}")
    
    # 统计结果
    avg_reward = np.mean(total_rewards)
    success_rate = success_episodes / 5.0
    
    print(f"\n📊 测试结果:")
    print(f"平均奖励: {avg_reward:.2f}")
    print(f"成功率: {success_rate:.1%}")
    print(f"碰撞次数: {collision_episodes}")
    
    env.close()

if __name__ == "__main__":
    # 设置随机种子以保证可重复性
    np.random.seed(42)
    torch.manual_seed(42)
    
    # 训练模型
    model = train_ppo_model()
    
    # 测试模型
    test_trained_model()
    
    print("\n🎉 训练和测试完成！")