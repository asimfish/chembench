#!/usr/bin/env python3
"""
测试 Pi0.5 远程策略服务器连接
随机构造输入数据，验证服务器能否正常响应
"""

import numpy as np
import time

# Pi0.5 远程推理客户端
from openpi_client import websocket_client_policy
from openpi_client import image_tools

# 服务器配置
PI0_SERVER_HOST = "81.68.132.224"
PI0_SERVER_PORT = 18015

def create_random_observation():
    """创建随机的观测数据（模拟仿真环境的输出）"""
    
    # 随机图像 (H, W, 3) uint8 格式
    # 模拟 224x224 的 RGB 图像
    head_img = np.random.randint(0, 256, size=(224, 224, 3), dtype=np.uint8)
    chest_img = np.random.randint(0, 256, size=(224, 224, 3), dtype=np.uint8)
    third_person_img = np.random.randint(0, 256, size=(224, 224, 3), dtype=np.uint8)
    
    # 随机状态向量 (33,) float32
    # 包含: arm2_pos(7) + arm2_vel(7) + hand2_pos(6) + hand2_vel(6) + eef_pos(3) + eef_quat(4) = 33
    state = np.random.randn(33).astype(np.float32)
    
    # 任务提示语
    prompt = "Pick up the green carton of drink from the table."
    
    return {
        "observation/head_camera": head_img,
        "observation/chest_camera": chest_img,
        "observation/third_person_camera": third_person_img,
        "observation/state": state,
        "prompt": prompt,
    }


def test_connection():
    """测试与 Pi0.5 服务器的连接"""
    
    print("=" * 60)
    print("Pi0.5 远程策略服务器连接测试")
    print("=" * 60)
    print(f"服务器地址: {PI0_SERVER_HOST}:{PI0_SERVER_PORT}")
    print()
    
    # Step 1: 尝试连接服务器
    print("[Step 1] 正在连接服务器...")
    try:
        start_time = time.time()
        client = websocket_client_policy.WebsocketClientPolicy(
            host=PI0_SERVER_HOST, 
            port=PI0_SERVER_PORT
        )
        connect_time = time.time() - start_time
        print(f"  ✅ 连接成功! 耗时: {connect_time:.3f}s")
    except Exception as e:
        print(f"  ❌ 连接失败: {e}")
        return False
    
    # Step 2: 获取服务器元数据
    print("\n[Step 2] 获取服务器元数据...")
    try:
        metadata = client.get_server_metadata()
        print(f"  ✅ 服务器元数据: {metadata}")
    except Exception as e:
        print(f"  ❌ 获取元数据失败: {e}")
    
    # Step 3: 构造随机输入并推理
    print("\n[Step 3] 发送随机观测数据进行推理...")
    try:
        observation = create_random_observation()
        print(f"  输入数据形状:")
        print(f"    - head_camera: {observation['observation/head_camera'].shape}")
        print(f"    - chest_camera: {observation['observation/chest_camera'].shape}")
        print(f"    - third_person_camera: {observation['observation/third_person_camera'].shape}")
        print(f"    - state: {observation['observation/state'].shape}")
        print(f"    - prompt: '{observation['prompt']}'")
        
        start_time = time.time()
        result = client.infer(observation, step=0)
        infer_time = time.time() - start_time
        
        print(f"\n  ✅ 推理成功! 耗时: {infer_time:.3f}s")
        print(f"  输出动作形状: {result['actions'].shape}")
        print(f"  动作范围: [{result['actions'].min():.4f}, {result['actions'].max():.4f}]")
        print(f"  动作示例 (第一步): {result['actions'][0]}")
        
    except Exception as e:
        print(f"  ❌ 推理失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 4: 多次推理测试延迟
    print("\n[Step 4] 测试多次推理延迟...")
    try:
        latencies = []
        for i in range(5):
            observation = create_random_observation()
            start_time = time.time()
            result = client.infer(observation, step=i)
            latency = time.time() - start_time
            latencies.append(latency)
            print(f"  推理 #{i+1}: {latency:.3f}s")
        
        avg_latency = np.mean(latencies)
        print(f"\n  平均延迟: {avg_latency:.3f}s")
        print(f"  预计帧率: {1/avg_latency:.1f} FPS (仅推理，不含仿真)")
        
    except Exception as e:
        print(f"  ❌ 多次推理测试失败: {e}")
    
    print("\n" + "=" * 60)
    print("✅ 测试完成! 服务器连接正常")
    print("=" * 60)
    return True


if __name__ == "__main__":
    test_connection()

