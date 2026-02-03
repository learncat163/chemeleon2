"""
单步精度测试 - 使用固定输入进行精确对比

用法示例：
    # 测试单个模块
    python test_single_step.py --module vae_encoder
    
    # 使用固定输入测试
    python test_single_step.py --module vae_encoder --use-fixed-input
    
    # 加载并对比两次测试结果
    python test_single_step.py --module vae_encoder --compare
"""

import sys
from pathlib import Path
import argparse

import numpy as np
import torch

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from test_precision import (
    set_seed, test_single_module, load_fixed_inputs,
    set_logger, close_logger
)


def compare_results(result1_path, result2_path, module_name):
    """对比两次测试的结果
    
    Args:
        result1_path: 第一次测试结果（如PyTorch）
        result2_path: 第二次测试结果（如Paddle）
        module_name: 要对比的模块名
    """
    print("\n" + "="*80)
    print(f"精度对比: {module_name}")
    print("="*80)
    
    # 加载结果
    data1 = np.load(result1_path, allow_pickle=True)
    data2 = np.load(result2_path, allow_pickle=True)
    
    if module_name not in data1 or module_name not in data2:
        print(f"❌ 模块 {module_name} 在结果文件中不存在")
        return
    
    module1 = data1[module_name].item()
    module2 = data2[module_name].item()
    
    # 对比每个输出
    print(f"\n对比 {result1_path.name} vs {result2_path.name}:\n")
    
    all_keys = set(module1.keys()) | set(module2.keys())
    
    for key in sorted(all_keys):
        if key == 'input':
            continue  # 跳过输入数据
            
        if key not in module1:
            print(f"  {key}: ⚠️  仅在结果2中存在")
            continue
        if key not in module2:
            print(f"  {key}: ⚠️  仅在结果1中存在")
            continue
        
        arr1 = module1[key]
        arr2 = module2[key]
        
        if arr1 is None or arr2 is None:
            print(f"  {key}: ⚠️  包含None值")
            continue
        
        if arr1.shape != arr2.shape:
            print(f"  {key}: ❌ 形状不匹配 {arr1.shape} vs {arr2.shape}")
            continue
        
        # 计算精度指标
        diff = np.abs(arr1 - arr2)
        mae = diff.mean()
        mse = np.square(diff).mean()
        max_diff = diff.max()
        
        # 相对误差
        arr1_abs_mean = np.abs(arr1).mean()
        if arr1_abs_mean > 1e-10:
            relative_error = mae / arr1_abs_mean
            rel_error_str = f"{relative_error:.2%}"
        else:
            rel_error_str = "N/A (值太小)"
        
        # 判断精度等级
        if mae < 1e-5:
            level = "✅ 优秀"
        elif mae < 1e-4:
            level = "✅ 良好"
        elif mae < 1e-3:
            level = "⚠️  一般"
        else:
            level = "❌ 差"
        
        print(f"  {key}:")
        print(f"    MAE: {mae:.6e}  MSE: {mse:.6e}  Max: {max_diff:.6e}")
        print(f"    相对误差: {rel_error_str}  {level}")
    
    print("\n精度等级说明:")
    print("  ✅ 优秀: MAE < 1e-5")
    print("  ✅ 良好: MAE < 1e-4")
    print("  ⚠️  一般: MAE < 1e-3")
    print("  ❌ 差:   MAE > 1e-3")


def main():
    parser = argparse.ArgumentParser(description="单步精度测试工具")
    parser.add_argument(
        "--module", 
        type=str,
        choices=['vae_encoder', 'vae_decoder', 'vae_full', 'ldm_denoiser', 'ldm_sampling'],
        default='vae_encoder',
        help="要测试的模块"
    )
    parser.add_argument(
        "--use-fixed-input",
        action="store_true",
        help="使用固定输入（从之前保存的结果加载）"
    )
    parser.add_argument(
        "--fixed-input-path",
        type=str,
        default="cyy_test/outputs/precision_test/pytorch_results.npz",
        help="固定输入数据的路径"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="cyy_test/outputs/precision_test/single_step_results.npz",
        help="输出文件路径"
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="对比两次测试结果"
    )
    parser.add_argument(
        "--compare-with",
        type=str,
        default="cyy_test/outputs/precision_test/pytorch_results.npz",
        help="要对比的另一个结果文件"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="设备 (cuda/cpu)"
    )
    
    args = parser.parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 模型路径
    vae_path = "checkpoints/v0.0.1/alex_mp_20/vae/dng_j1jgz9t0_v1.ckpt"
    ldm_path = "checkpoints/v0.0.1/alex_mp_20/ldm/ldm_rl_dng_tuor5vgd.ckpt"
    structure_path = "cyy_test/outputs/test_three/generated_structures.json.gz"
    
    if not Path(structure_path).exists():
        structure_path = "outputs/alex-mp-20/generated_structures.json.gz"
    
    # 设备
    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    
    print("\n" + "🔬 " * 40)
    print(" " * 30 + "单步精度测试")
    print("🔬 " * 40)
    print(f"\n模块: {args.module}")
    print(f"随机种子: {args.seed}")
    print(f"设备: {device}")
    print(f"使用固定输入: {args.use_fixed_input}")
    
    # 对比模式
    if args.compare:
        output_path = Path(args.output)
        compare_with_path = Path(args.compare_with)
        
        if not output_path.exists():
            print(f"\n❌ 结果文件不存在: {output_path}")
            return False
        if not compare_with_path.exists():
            print(f"\n❌ 对比文件不存在: {compare_with_path}")
            return False
        
        compare_results(output_path, compare_with_path, args.module)
        return True
    
    # 测试模式
    output_dir = Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 设置日志
    log_file = output_dir / f"single_step_{args.module}.log"
    set_logger(log_file)
    
    try:
        # 加载固定输入
        fixed_inputs = None
        if args.use_fixed_input:
            fixed_input_path = Path(args.fixed_input_path)
            if fixed_input_path.exists():
                print(f"\n✓ 从 {fixed_input_path} 加载固定输入")
                fixed_inputs = load_fixed_inputs(fixed_input_path)
            else:
                print(f"\n⚠️  固定输入文件不存在: {fixed_input_path}")
                print("   将使用随机输入")
        
        # 运行测试
        result = test_single_module(
            args.module,
            vae_path,
            ldm_path,
            structure_path,
            device,
            args.use_fixed_input,
            fixed_inputs
        )
        
        # 保存结果
        output_path = Path(args.output)
        np.savez(output_path, **{args.module: result})
        
        print("\n" + "="*80)
        print("测试完成！")
        print("="*80)
        print(f"\n✓ 结果已保存至: {output_path}")
        print(f"✓ 日志已保存至: {log_file}")
        
        # 显示如何对比
        print("\n💡 进行精度对比:")
        print(f"   python cyy_test/test_single_step.py --module {args.module} --compare")
        
        close_logger()
        return True
        
    except Exception as e:
        print(f"\n❌ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        close_logger()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
