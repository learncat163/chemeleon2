#!/usr/bin/env python3
"""
示例：如何加载和使用test_precision.py生成的NPZ文件

这个脚本展示如何：
1. 加载JSON摘要文件
2. 从JSON中获取NPZ文件路径
3. 加载完整的tensor数据
4. 对比不同框架的结果
"""

import numpy as np
import json
from pathlib import Path


def load_test_results(json_path):
    """加载测试结果的JSON文件"""
    with open(json_path) as f:
        return json.load(f)


def load_full_tensor(npz_path):
    """从NPZ文件加载完整的tensor数据"""
    data = np.load(npz_path)
    return data['data']


def compare_tensors(tensor1, tensor2, name=""):
    """对比两个tensor的差异"""
    diff = np.abs(tensor1 - tensor2)
    rel_diff = diff / (np.abs(tensor1) + 1e-8)
    
    print(f"\n{'='*60}")
    print(f"对比: {name}")
    print(f"{'='*60}")
    print(f"形状: {tensor1.shape} vs {tensor2.shape}")
    print(f"\n绝对误差:")
    print(f"  最大值: {diff.max():.2e}")
    print(f"  平均值: {diff.mean():.2e}")
    print(f"  中位数: {np.median(diff):.2e}")
    print(f"\n相对误差:")
    print(f"  最大值: {rel_diff.max():.2e}")
    print(f"  平均值: {rel_diff.mean():.2e}")
    print(f"\n统计对比:")
    print(f"  均值: {tensor1.mean():.6f} vs {tensor2.mean():.6f}")
    print(f"  标准差: {tensor1.std():.6f} vs {tensor2.std():.6f}")
    print(f"  最小值: {tensor1.min():.6f} vs {tensor2.min():.6f}")
    print(f"  最大值: {tensor1.max():.6f} vs {tensor2.max():.6f}")


def example_1_basic_load():
    """示例1：基本的加载和查看"""
    print("\n" + "="*60)
    print("示例 1: 基本的NPZ文件加载")
    print("="*60)
    
    # 加载JSON摘要
    json_path = "cyy_test/outputs/precision_test/pytorch_results.json"
    results = load_test_results(json_path)
    
    print(f"\n测试信息:")
    print(f"  框架: {results['test_info']['framework']}")
    print(f"  测试时间: {results['test_info']['test_time']}")
    print(f"  随机种子: {results['test_info']['random_seed']}")
    print(f"  设备: {results['test_info']['device']}")
    
    # 获取VAE编码器的latent_mean信息
    latent_mean_info = results['modules']['vae_encoder']['output']['latent_mean']
    print(f"\nVAE Encoder Latent Mean (JSON摘要):")
    print(f"  形状: {latent_mean_info['shape']}")
    print(f"  数据类型: {latent_mean_info['dtype']}")
    print(f"  前4个值: {latent_mean_info['first_4_values']}")
    print(f"  统计: mean={latent_mean_info['mean']:.6f}, std={latent_mean_info['std']:.6f}")
    print(f"  NPZ文件: {latent_mean_info['full_data_file']}")
    
    # 加载完整数据
    npz_path = latent_mean_info['full_data_file']
    if Path(npz_path).exists():
        full_tensor = load_full_tensor(npz_path)
        print(f"\n完整数据 (从NPZ加载):")
        print(f"  形状: {full_tensor.shape}")
        print(f"  数据类型: {full_tensor.dtype}")
        print(f"  前4个值: {full_tensor.flatten()[:4]}")
        print(f"  统计: mean={full_tensor.mean():.6f}, std={full_tensor.std():.6f}")
        
        # 验证一致性
        json_first_4 = np.array(latent_mean_info['first_4_values'])
        npz_first_4 = full_tensor.flatten()[:4]
        is_consistent = np.allclose(json_first_4, npz_first_4, rtol=1e-5)
        print(f"\n✓ JSON和NPZ数据一致性: {'通过' if is_consistent else '失败'}")
    else:
        print(f"\n✗ NPZ文件不存在: {npz_path}")


def example_2_compare_frameworks():
    """示例2：对比两个框架的结果"""
    print("\n" + "="*60)
    print("示例 2: 对比PyTorch和Paddle的结果")
    print("="*60)
    
    # 假设你有两个框架的结果
    pytorch_json = "cyy_test/outputs/precision_test/pytorch_results.json"
    # paddle_json = "cyy_test/outputs/precision_test/paddle_results.json"  # 假设的路径
    
    if not Path(pytorch_json).exists():
        print(f"\n✗ PyTorch结果不存在: {pytorch_json}")
        return
    
    # 加载PyTorch结果
    pytorch_results = load_test_results(pytorch_json)
    pytorch_latent_mean_path = pytorch_results['modules']['vae_encoder']['output']['latent_mean']['full_data_file']
    
    if Path(pytorch_latent_mean_path).exists():
        pytorch_tensor = load_full_tensor(pytorch_latent_mean_path)
        print(f"\n✓ 成功加载PyTorch的latent_mean:")
        print(f"  形状: {pytorch_tensor.shape}")
        print(f"  统计: mean={pytorch_tensor.mean():.6f}, std={pytorch_tensor.std():.6f}")
        
        # 如果有Paddle的结果，可以这样对比：
        # paddle_results = load_test_results(paddle_json)
        # paddle_latent_mean_path = paddle_results['modules']['vae_encoder']['output']['latent_mean']['full_data_file']
        # paddle_tensor = load_full_tensor(paddle_latent_mean_path)
        # compare_tensors(pytorch_tensor, paddle_tensor, "VAE Encoder Latent Mean")
        
        print("\n💡 提示: 当你有Paddle结果时，取消注释上面的代码进行对比")
    else:
        print(f"\n✗ PyTorch NPZ文件不存在: {pytorch_latent_mean_path}")


def example_3_iterate_all_outputs():
    """示例3：遍历所有模块的所有输出"""
    print("\n" + "="*60)
    print("示例 3: 遍历所有模块的输出")
    print("="*60)
    
    json_path = "cyy_test/outputs/precision_test/pytorch_results.json"
    if not Path(json_path).exists():
        print(f"\n✗ 结果文件不存在: {json_path}")
        return
    
    results = load_test_results(json_path)
    
    for module_name, module_data in results['modules'].items():
        print(f"\n{'='*60}")
        print(f"模块: {module_name}")
        print(f"描述: {module_data['description']}")
        print(f"{'='*60}")
        
        if 'output' in module_data:
            for output_name, output_info in module_data['output'].items():
                if isinstance(output_info, dict) and 'full_data_file' in output_info:
                    npz_path = output_info['full_data_file']
                    exists = "✓" if Path(npz_path).exists() else "✗"
                    print(f"  {exists} {output_name}: {npz_path}")
                    if 'shape' in output_info:
                        print(f"      形状: {output_info['shape']}, 元素数: {output_info['total_elements']}")


if __name__ == "__main__":
    print("\n" + "🔬 " * 30)
    print(" " * 20 + "NPZ文件使用示例")
    print("🔬 " * 30)
    
    # 运行所有示例
    example_1_basic_load()
    example_2_compare_frameworks()
    example_3_iterate_all_outputs()
    
    print("\n" + "="*60)
    print("示例完成！")
    print("="*60)
    print("\n💡 提示:")
    print("  1. JSON文件包含摘要信息（前4个值、统计量、NPZ路径）")
    print("  2. NPZ文件包含完整的tensor数据")
    print("  3. 使用JSON快速查看，使用NPZ进行精确对比")
    print("  4. 所有NPZ文件的数据存储在'data'键下")
    print()
