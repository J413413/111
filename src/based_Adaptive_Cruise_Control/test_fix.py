"""
测试修复脚本 - 验证环境兼容性修复是否成功
"""

import numpy as np
import time
from acc_env import ACCEnv

def test_environment_basic():
    """
    测试环境基本功能
    """
    print("🧪 测试环境基本功能")
    print("-" * 50)
    
    try:
        # 创建环境
        env = ACCEnv(render_mode=None)
        
        # 测试重置功能
        obs, info = env.reset(seed=42)
        print(f"✅ 环境重置成功")
        print(f"  初始观测值: {obs}")
        print(f"  信息字典: {info}")
        
        # 测试步数功能
        action = np.array([0.5], dtype=np.float32)  # 加速度为0.5 m/s²
        obs, reward, terminated, truncated, info = env.step(action)
        
        print(f"✅ 单步执行成功")
        print(f"  新观测值: {obs}")
        print(f"  奖励: {reward:.2f}")
        print(f"  终止状态: {terminated}, {truncated}")
        print(f"  信息: {info}")
        
        # 测试多个步数
        print("\n🔄 测试10个连续步数...")
        total_reward = 0
        for i in range(10):
            action = np.array([np.random.uniform(-1.0, 1.0)], dtype=np.float32)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            if terminated or truncated:
                print(f"❌ 提前终止在第 {i+1} 步")
                break
        
        print(f"✅ 连续步数测试完成")
        print(f"  总奖励: {total_reward:.2f}")
        print(f"  最终速度: {info['ego_speed']:.2f} m/s")
        print(f"  最终距离: {info['distance']:.2f} m")
        
        env.close()
        return True
        
    except Exception as e:
        print(f"❌ 环境测试失败: {str(e)}")
        return False

def test_environment_reset():
    """
    测试环境重置功能
    """
    print("\n🔄 测试环境重置功能")
    print("-" * 50)
    
    try:
        env = ACCEnv(render_mode=None)
        
        # 第一次重置
        obs1, _ = env.reset(seed=42)
        
        # 执行一些步数
        for _ in range(5):
            action = np.array([0.5], dtype=np.float32)
            env.step(action)
        
        # 第二次重置
        obs2, _ = env.reset(seed=42)
        
        # 验证两次重置的结果是否相同（相同种子）
        if np.allclose(obs1, obs2, atol=1e-6):
            print("✅ 环境重置功能正常 - 相同种子产生相同初始状态")
        else:
            print("⚠️  警告: 相同种子产生不同初始状态")
            print(f"  第一次: {obs1}")
            print(f"  第二次: {obs2}")
        
        env.close()
        return True
        
    except Exception as e:
        print(f"❌ 重置测试失败: {str(e)}")
        return False

def test_environment_rendering():
    """
    测试环境渲染功能
    """
    print("\n🎨 测试环境渲染功能")
    print("-" * 50)
    
    try:
        env = ACCEnv(render_mode='human')
        
        # 测试渲染
        obs, _ = env.reset()
        env.render()
        
        print("✅ 渲染功能测试完成")
        
        env.close()
        return True
        
    except Exception as e:
        print(f"❌ 渲染测试失败: {str(e)}")
        return False

def test_environment_compatibility():
    """
    测试环境与Gymnasium的兼容性
    """
    print("\n🔗 测试Gymnasium兼容性")
    print("-" * 50)
    
    try:
        import gymnasium as gym
        
        # 测试环境注册
        env_id = 'ACC-v0'
        if env_id in gym.envs.registry:
            print(f"✅ 环境 '{env_id}' 已正确注册")
        else:
            print(f"❌ 环境 '{env_id}' 未注册")
            return False
        
        # 通过注册ID创建环境
        env = gym.make(env_id)
        obs, info = env.reset(seed=42)
        
        print(f"✅ 通过gym.make()创建环境成功")
        print(f"  观测空间: {env.observation_space}")
        print(f"  动作空间: {env.action_space}")
        print(f"  初始观测值: {obs}")
        
        env.close()
        return True
        
    except Exception as e:
        print(f"❌ 兼容性测试失败: {str(e)}")
        return False

def test_performance():
    """
    测试环境性能
    """
    print("\n⚡ 测试环境性能")
    print("-" * 50)
    
    try:
        env = ACCEnv(render_mode=None)
        
        # 测试1000步的执行时间
        num_steps = 1000
        
        start_time = time.time()
        
        obs, _ = env.reset()
        for i in range(num_steps):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            if terminated or truncated:
                obs, _ = env.reset()
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"✅ 性能测试完成")
        print(f"  执行 {num_steps} 步耗时: {duration:.2f} 秒")
        print(f"  每秒步数: {num_steps/duration:.1f} FPS")
        
        env.close()
        return True
        
    except Exception as e:
        print(f"❌ 性能测试失败: {str(e)}")
        return False

def main():
    """
    主函数
    """
    print("🚗 RL自适应巡航控制 - 环境修复测试")
    print("=" * 60)
    
    # 运行所有测试
    tests = [
        test_environment_basic,
        test_environment_reset,
        test_environment_rendering,
        test_environment_compatibility,
        test_performance
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
    print("📊 测试结果汇总")
    print("=" * 60)
    
    passed = sum(results)
    total = len(results)
    
    print(f"通过测试: {passed}/{total}")
    print(f"成功率: {passed/total*100:.1f}%")
    
    if passed == total:
        print("\n🎉 所有测试通过！环境修复成功！")
        print("\n下一步建议:")
        print("  1. 运行 'python train.py' 开始训练模型")
        print("  2. 运行 'python example_usage.py' 查看示例演示")
    else:
        print("\n⚠️  部分测试失败，建议检查错误信息并进一步修复")

if __name__ == "__main__":
    main()