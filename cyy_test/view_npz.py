#!/usr/bin/env python3
"""
NPZ 文件查看工具 - 人类可读的方式查看推理数据

用法:
    python view_npz.py <npz文件路径>
    python view_npz.py <npz文件路径> --key vae_encoder
    python view_npz.py <npz文件路径> --detail
"""

import sys
import argparse
from pathlib import Path
import numpy as np


def format_array_info(arr, name="", indent=0, show_values=False, max_values=10):
    """格式化数组信息"""
    prefix = "  " * indent
    
    if arr is None:
        return f"{prefix}{name}: None"
    
    lines = []
    lines.append(f"{prefix}{name}:")
    lines.append(f"{prefix}  类型: {arr.dtype}")
    lines.append(f"{prefix}  形状: {arr.shape}")
    lines.append(f"{prefix}  大小: {arr.size} 个元素 ({arr.nbytes / 1024:.2f} KB)")
    
    if arr.size > 0:
        lines.append(f"{prefix}  统计:")
        lines.append(f"{prefix}    最小值: {arr.min():.6e}" if arr.size else "    (空数组)")
        lines.append(f"{prefix}    最大值: {arr.max():.6e}" if arr.size else "")
        lines.append(f"{prefix}    均值:   {arr.mean():.6e}" if arr.size else "")
        lines.append(f"{prefix}    标准差: {arr.std():.6e}" if arr.size else "")
        
        if show_values:
            flat = arr.flatten()
            n_show = min(max_values, len(flat))
            lines.append(f"{prefix}  前 {n_show} 个值:")
            
            # 格式化输出，每行5个值
            for i in range(0, n_show, 5):
                end = min(i + 5, n_show)
                values = flat[i:end]
                value_str = "  ".join(f"{v:12.6e}" for v in values)
                lines.append(f"{prefix}    [{i:4d}]: {value_str}")
    
    return "\n".join(lines)


def view_npz_simple(npz_path, key=None):
    """简单查看模式"""
    print(f"\n📦 NPZ 文件: {npz_path}")
    print("=" * 80)
    
    data = np.load(npz_path, allow_pickle=True)
    
    print(f"\n包含的模块: {list(data.keys())}")
    print(f"文件大小: {Path(npz_path).stat().st_size / 1024:.2f} KB")
    
    # 如果指定了key，只查看该模块
    if key:
        if key not in data:
            print(f"\n❌ 模块 '{key}' 不存在")
            print(f"可用模块: {list(data.keys())}")
            return
        keys_to_view = [key]
    else:
        keys_to_view = data.keys()
    
    for module_name in keys_to_view:
        print(f"\n{'='*80}")
        print(f"模块: {module_name}")
        print('='*80)
        
        module_data = data[module_name].item() if data[module_name].dtype == object else data[module_name]
        
        if isinstance(module_data, dict):
            for sub_key, sub_value in module_data.items():
                if isinstance(sub_value, np.ndarray):
                    print(f"\n{format_array_info(sub_value, sub_key, indent=1)}")
                else:
                    print(f"  {sub_key}: {type(sub_value).__name__}")
        else:
            print(f"\n{format_array_info(module_data, 'data', indent=1)}")


def view_npz_detail(npz_path, key=None, max_values=20):
    """详细查看模式（显示具体数值）"""
    print(f"\n📦 NPZ 文件详细信息: {npz_path}")
    print("=" * 80)
    
    data = np.load(npz_path, allow_pickle=True)
    
    print(f"\n包含的模块: {list(data.keys())}")
    print(f"文件大小: {Path(npz_path).stat().st_size / 1024:.2f} KB")
    
    # 如果指定了key，只查看该模块
    if key:
        if key not in data:
            print(f"\n❌ 模块 '{key}' 不存在")
            return
        keys_to_view = [key]
    else:
        keys_to_view = data.keys()
    
    for module_name in keys_to_view:
        print(f"\n{'='*80}")
        print(f"模块: {module_name}")
        print('='*80)
        
        module_data = data[module_name].item() if data[module_name].dtype == object else data[module_name]
        
        if isinstance(module_data, dict):
            for sub_key, sub_value in module_data.items():
                if isinstance(sub_value, np.ndarray):
                    print(f"\n{format_array_info(sub_value, sub_key, indent=1, show_values=True, max_values=max_values)}")
                elif sub_value is None:
                    print(f"\n  {sub_key}: None")
                else:
                    print(f"\n  {sub_key}: {type(sub_value).__name__}")
        else:
            print(f"\n{format_array_info(module_data, 'data', indent=1, show_values=True, max_values=max_values)}")


def compare_two_npz(npz_path1, npz_path2, key=None):
    """对比两个npz文件"""
    print(f"\n🔍 对比 NPZ 文件")
    print("=" * 80)
    print(f"文件1: {npz_path1}")
    print(f"文件2: {npz_path2}")
    
    data1 = np.load(npz_path1, allow_pickle=True)
    data2 = np.load(npz_path2, allow_pickle=True)
    
    keys1 = set(data1.keys())
    keys2 = set(data2.keys())
    
    print(f"\n共同模块: {keys1 & keys2}")
    print(f"仅在文件1: {keys1 - keys2}")
    print(f"仅在文件2: {keys2 - keys1}")
    
    # 选择要对比的模块
    if key:
        if key not in keys1 or key not in keys2:
            print(f"\n❌ 模块 '{key}' 不在两个文件中都存在")
            return
        keys_to_compare = [key]
    else:
        keys_to_compare = keys1 & keys2
    
    for module_name in keys_to_compare:
        print(f"\n{'='*80}")
        print(f"对比模块: {module_name}")
        print('='*80)
        
        module1 = data1[module_name].item() if data1[module_name].dtype == object else data1[module_name]
        module2 = data2[module_name].item() if data2[module_name].dtype == object else data2[module_name]
        
        if isinstance(module1, dict) and isinstance(module2, dict):
            all_keys = set(module1.keys()) | set(module2.keys())
            
            for sub_key in sorted(all_keys):
                if sub_key not in module1:
                    print(f"\n  {sub_key}: ⚠️  仅在文件2中")
                    continue
                if sub_key not in module2:
                    print(f"\n  {sub_key}: ⚠️  仅在文件1中")
                    continue
                
                arr1 = module1[sub_key]
                arr2 = module2[sub_key]
                
                if arr1 is None or arr2 is None:
                    print(f"\n  {sub_key}: ⚠️  包含None")
                    continue
                
                if not isinstance(arr1, np.ndarray) or not isinstance(arr2, np.ndarray):
                    print(f"\n  {sub_key}: 非数组类型")
                    continue
                
                if arr1.shape != arr2.shape:
                    print(f"\n  {sub_key}: ❌ 形状不同")
                    print(f"    文件1: {arr1.shape}")
                    print(f"    文件2: {arr2.shape}")
                    continue
                
                diff = np.abs(arr1 - arr2)
                mae = diff.mean()
                max_diff = diff.max()
                
                if arr1.size > 0:
                    relative = mae / (np.abs(arr1).mean() + 1e-10)
                else:
                    relative = 0
                
                if mae < 1e-5:
                    status = "✅ 优秀"
                elif mae < 1e-4:
                    status = "✅ 良好"
                elif mae < 1e-3:
                    status = "⚠️  一般"
                else:
                    status = "❌ 差"
                
                print(f"\n  {sub_key}:")
                print(f"    MAE:      {mae:.6e}")
                print(f"    Max Diff: {max_diff:.6e}")
                print(f"    相对误差: {relative:.4%}")
                print(f"    状态:     {status}")


def export_to_text(npz_path, output_path, key=None):
    """导出为文本文件"""
    data = np.load(npz_path, allow_pickle=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(f"NPZ 文件: {npz_path}\n")
        f.write("=" * 80 + "\n\n")
        
        keys_to_export = [key] if key and key in data else data.keys()
        
        for module_name in keys_to_export:
            f.write(f"\n模块: {module_name}\n")
            f.write("=" * 80 + "\n")
            
            module_data = data[module_name].item() if data[module_name].dtype == object else data[module_name]
            
            if isinstance(module_data, dict):
                for sub_key, sub_value in module_data.items():
                    if isinstance(sub_value, np.ndarray):
                        f.write(f"\n{sub_key}:\n")
                        f.write(f"  形状: {sub_value.shape}\n")
                        f.write(f"  类型: {sub_value.dtype}\n")
                        if sub_value.size > 0:
                            f.write(f"  数据:\n")
                            np.savetxt(f, sub_value.reshape(-1, sub_value.shape[-1]) if sub_value.ndim > 1 else sub_value.reshape(-1, 1), 
                                      fmt='%.6e', delimiter='\t')
                        f.write("\n")
    
    print(f"\n✅ 已导出到: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="NPZ 文件查看工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 查看文件概览
  python view_npz.py results.npz
  
  # 查看特定模块
  python view_npz.py results.npz --key vae_encoder
  
  # 详细查看（显示数值）
  python view_npz.py results.npz --detail
  
  # 对比两个文件
  python view_npz.py results1.npz --compare results2.npz
  
  # 导出为文本
  python view_npz.py results.npz --export output.txt
        """
    )
    
    parser.add_argument('npz_file', help='NPZ 文件路径')
    parser.add_argument('--key', '-k', help='只查看指定的模块')
    parser.add_argument('--detail', '-d', action='store_true', help='详细模式（显示具体数值）')
    parser.add_argument('--compare', '-c', help='对比另一个 NPZ 文件')
    parser.add_argument('--export', '-e', help='导出为文本文件')
    parser.add_argument('--max-values', '-m', type=int, default=20, help='显示的最大数值数量（默认20）')
    
    args = parser.parse_args()
    
    npz_path = Path(args.npz_file)
    if not npz_path.exists():
        print(f"❌ 文件不存在: {npz_path}")
        return 1
    
    try:
        if args.compare:
            compare_path = Path(args.compare)
            if not compare_path.exists():
                print(f"❌ 对比文件不存在: {compare_path}")
                return 1
            compare_two_npz(npz_path, compare_path, args.key)
        elif args.export:
            export_to_text(npz_path, args.export, args.key)
        elif args.detail:
            view_npz_detail(npz_path, args.key, args.max_values)
        else:
            view_npz_simple(npz_path, args.key)
        
        return 0
    
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
