"""
修复验证脚本 - 全面测试所有修复是否成功
"""

import os
import sys
import numpy as np
import torch
import time
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from acc_env import ACCEnv

def test_environment_setup():
    """
    测试环境设置
    """
    print("🧪 测试环境设置")
    print("-" * 50)
    
    try:
        # 导入检查
        import gymnasium as gym
        print(f"✅ 成功导入 gymnasium {gym.__version__}")
        
        # 检查环境注册
        env_id = 'ACC-v0'
        if env_id in gym.envs.registry:
            print(f"✅ 环境 '{env_id}' 已正确注册")
        else:
            print(f"❌ 环境 '{env_id}' 未注册")
            return False
        
        # 创建环境
        env = gym.make(env_id, render_mode=None)
        print(f"✅ 通过 gym.make() 创建环境成功")
        
        # 检查空间类型
        print(f"  观测空间类型: {type(env.observation_space)}")
        print(f"  动作空间类型: {type(env.action_space)}")
        
        # 检查空间维度
        print(f"  观测空间形状: {env.observation_space.shape}")
        print(f"  动作空间形状: {env.action_space.shape}")
        
        # 检查数据类型
        print(f"  观测空间数据类型: {env.observation_space.dtype}")
        print(f"  动作空间数据类型: {env.action_space.dtype}")
        
        env.close()
        return True
        
    except Exception as e:
        print(f"❌ 环境设置测试失败: {str(e)}")
        return False

def test_environment_dynamics():
    """
    测试环境动力学
    """
    print("\n⚡ 测试环境动力学")
    print("-" * 50)
    
    try:
        env = ACCEnv(render_mode=None)
        
        # 测试重置和步数
        obs, info = env.reset(seed=42)
        print(f"✅ 环境重置成功，观测值形状: {obs.shape}")
        
        # 测试多个动作
        actions = [
            np.array([0.0], dtype=np.float32),   # 零加速度
            np.array([1.0], dtype=np.float32),   # 正加速度
            np.array([-1.0], dtype=np.float32),  # 负加速度
            np.array([2.0], dtype=np.float32),   # 最大加速度
            np.array([-3.0], dtype=np.float32)   # 最大减速度
        ]
        
        for i, action in enumerate(actions):
            obs, reward, terminated, truncated, info = env.step(action)
            
            print(f"✅ 动作 {i+1} 执行成功")
            print(f"  动作: {action[0]:.1f} m/s²")
            print(f"  奖励: {reward:.2f}")
            print(f"  自车速度: {info['ego_speed']:.2f} m/s")
            print(f"  车间距离: {info['distance']:.2f} m")
            
            if terminated or truncated:
                print(f"  ⚠️  回合结束")
                obs, info = env.reset()
        
        env.close()
        return True
        
    except Exception as e:
        print(f"❌ 环境动力学测试失败: {str(e)}")
        return False

def test_model_creation():
    """
    测试模型创建
    """
    print("\n🤖 测试PPO模型创建")
    print("-" * 50)
    
    try:
        # 创建环境
        env = ACCEnv(render_mode=None)
        
        # 测试不同的策略参数配置
        policy_configs = [
            {
                'name': '默认配置',
                'config': {}
            },
            {
                'name': '自定义网络架构',
                'config': {
                    'net_arch': [64, 64]
                }
            },
            {
                'name': '不同激活函数',
                'config': {
                    'activation_fn': torch.nn.Tanh
                }
            }
        ]
        
        for config in policy_configs:
            print(f"\n📋 测试 {config['name']}")
            
            try:
                model = PPO(
                    "MlpPolicy",
                    env,
                    verbose=0,
                    learning_rate=3e-4,
                    n_steps=512,
                    batch_size=64,
                    n_epochs=4,
                    gamma=0.99,
                    policy_kwargs=config['config']
                )
                
                print(f"  ✅ 模型创建成功")
                print(f"    策略网络: {model.policy}")
                
                # 清理模型
                del model
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
            except Exception as e:
                print(f"  ❌ 失败: {str(e)}")
                return False
        
        env.close()
        return True
        
    except Exception as e:
        print(f"❌ 模型创建测试失败: {str(e)}")
        return False

def test_training_functionality():
    """
    测试训练功能
    """
    print("\n🎯 测试训练功能")
    print("-" * 50)
    
    try:
        # 创建环境
        env = make_vec_env(lambda: ACCEnv(render_mode=None), n_envs=1)
        
        # 创建简单模型
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=3e-4,
            n_steps=256,
            batch_size=32,
            n_epochs=2,
            gamma=0.99,
            policy_kwargs={
                'net_arch': [32, 32],
                'activation_fn': torch.nn.ReLU
            }
        )
        
        print("✅ 开始短时间训练测试...")
        
        # 记录开始时间
        start_time = time.time()
        
        # 训练一小段时间
        model.learn(total_timesteps=1000, log_interval=1)
        
        # 计算训练时间
        training_time = time.time() - start_time
        
        print(f"✅ 训练完成！耗时: {training_time:.2f} 秒")
        
        # 保存模型
        model_path = "models/verification_model"
        model.save(model_path)
        print(f"✅ 模型保存成功: {model_path}.zip")
        
        # 加载模型
        loaded_model = PPO.load(model_path)
        print(f"✅ 模型加载成功")
        
        # 测试预测
        obs = env.reset()
        action, _ = loaded_model.predict(obs, deterministic=True)
        print(f"✅ 模型预测成功: {action}")
        
        # 清理
        env.close()
        del model, loaded_model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        return True
        
    except Exception as e:
        print(f"❌ 训练功能测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_full_pipeline():
    """
    测试完整流程
    """
    print("\n🔄 测试完整训练和评估流程")
    print("-" * 50)
    
    try:
        # 创建环境
        env = ACCEnv(render_mode=None)
        
        # 创建模型
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=3e-4,
            n_steps=512,
            batch_size=64,
            n_epochs=4,
            gamma=0.99,
            policy_kwargs={
                'net_arch': [64, 64]
            }
        )
        
        # 训练
        print("🚀 开始训练...")
        model.learn(total_timesteps=2000, log_interval=1)
        
        # 保存模型
        model.save("models/full_pipeline_model")
        
        # 评估
        print("\n📊 开始评估...")
        obs, _ = env.reset()
        total_reward = 0
        done = False
        steps = 0
        
        while not done and steps < 100:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            
            done = terminated or truncated
        
        print(f"✅ 评估完成")
        print(f"  总步数: {steps}")
        print(f"  总奖励: {total_reward:.2f}")
        print(f"  平均奖励: {total_reward/steps:.3f}")
        
        env.close()
        return True
        
    except Exception as e:
        print(f"❌ 完整流程测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """
    主函数
    """
    print("🚗 RL自适应巡航控制 - 修复验证")
    print("=" * 60)
    
    # 创建必要的目录
    os.makedirs("models", exist_ok=True)
    
    # 运行所有测试
    tests = [
        test_environment_setup,
        test_environment_dynamics,
        test_model_creation,
        test_training_functionality,
        test_full_pipeline
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"\n❌ 测试 {test.__name__} 异常: {str(e)}")
            results.append(False)
    
    # 汇总结果
    print("\n" + "=" * 60)
    print("📋 修复验证结果")
    print("=" * 60)
    
    passed = sum(results)
    total = len(results)
    
    print(f"通过测试: {passed}/{total}")
    print(f"成功率: {passed/total*100:.1f}%")
    
    if passed == total:
        print("\n🎉 所有修复验证通过！项目现在应该可以正常工作了。")
        print("\n🚀 建议的下一步操作:")
        print("  1. 运行完整训练: python train.py")
        print("  2. 测试训练模型: python test.py")
        print("  3. 查看示例演示: python example_usage.py")
    else:
        print("\n⚠️  部分测试失败，建议进一步检查和修复。")
        print("\n📝 失败的测试:")
        for i, (test, result) in enumerate(zip(tests, results)):
            if not result:
                print(f"  - {test.__name__}")

if __name__ == "__main__":
    main()