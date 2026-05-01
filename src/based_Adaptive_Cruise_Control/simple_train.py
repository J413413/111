"""
简化的训练脚本 - 用于验证修复是否成功
"""

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from acc_env import ACCEnv
from config import TARGET_SPEED, SAFETY_DISTANCE

def main():
    """
    主函数 - 简化的训练流程
    """
    print("🚗 简化训练脚本 - 验证修复")
    print("=" * 50)
    
    try:
        # 设置随机种子
        np.random.seed(42)
        torch.manual_seed(42)
        
        # 创建环境
        print("📦 创建环境...")
        env = ACCEnv(render_mode=None)
        
        # 验证环境空间
        print(f"✅ 观测空间: {env.observation_space}")
        print(f"✅ 动作空间: {env.action_space}")
        
        # 测试环境基本功能
        print("\n🧪 测试环境功能...")
        obs, info = env.reset(seed=42)
        print(f"  初始观测值: {obs}")
        
        action = np.array([0.5], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"  单步奖励: {reward:.2f}")
        
        # 创建PPO模型（使用简化参数）
        print("\n🤖 创建PPO模型...")
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=3e-4,
            n_steps=1024,
            batch_size=64,
            n_epochs=4,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.0,
            vf_coef=0.5,
            max_grad_norm=0.5,
            use_sde=False,
            policy_kwargs={
                'net_arch': [64, 64],
                'activation_fn': torch.nn.ReLU,
                'ortho_init': True,
                'log_std_init': 0.0,
                'full_std': True,
                'squash_output': False,
                'share_features_extractor': False
            }
        )
        
        print("✅ PPO模型创建成功！")
        
        # 进行短时间训练测试
        print("\n🎯 开始训练测试（1000步）...")
        model.learn(total_timesteps=1000, log_interval=1)
        
        print("✅ 训练测试完成！")
        
        # 保存模型
        model.save("models/test_model")
        print("💾 模型已保存到: models/test_model.zip")
        
        # 测试保存的模型
        print("\n🧪 测试保存的模型...")
        loaded_model = PPO.load("models/test_model")
        
        # 测试模型预测
        obs, _ = env.reset()
        action, _ = loaded_model.predict(obs, deterministic=True)
        print(f"  模型预测动作: {action}")
        
        print("✅ 模型加载和预测测试成功！")
        
        env.close()
        print("\n🎉 所有测试通过！修复成功！")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()