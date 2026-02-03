#!/usr/bin/env python3
"""
精度对比工具 - 对比两个测试结果（支持JSON和TXT格式）

用法:
    python compare_precision.py pytorch_results.json paddle_results.json
    python compare_precision.py pytorch_results.json paddle_results.json --module vae_encoder
"""

import sys
import json
import argparse
from pathlib import Path


def parse_results_file(filepath):
    """解析结果文件（支持JSON和TXT格式）"""
    filepath = Path(filepath)
    
    # 如果是JSON文件
    if filepath.suffix == '.json':
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 提取模块数据
        if 'modules' in data:
            return data['modules']
        else:
            return data
    
    # 如果是TXT文件（旧格式）
    results = {}
    current_module = None
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            
            # 检测模块开始
            if line.startswith("模块:"):
                current_module = line.split(":", 1)[1].strip()
                results[current_module] = {}
            
            # 解析键值对
            elif ":" in line and current_module and not line.startswith("="):
                parts = line.split(":", 1)
                if len(parts) == 2:
                    key = parts[0].strip()
                    value = parts[1].strip()
                    results[current_module][key] = value
    
    return results


def compare_values(val1, val2, key):
    """对比两个值（支持字符串、数字、列表）"""
    # 如果是列表（JSON格式中的first_4_values）
    if isinstance(val1, list) and isinstance(val2, list):
        if len(val1) != len(val2):
            return f"❌ 数量不同 ({len(val1)} vs {len(val2)})"
        
        diffs = [abs(float(v1) - float(v2)) for v1, v2 in zip(val1, val2)]
        max_diff = max(diffs)
        avg_diff = sum(diffs) / len(diffs)
        
        # 相对误差
        avg_val = sum(abs(float(v)) for v in val1) / len(val1)
        rel_error = avg_diff / (avg_val + 1e-10)
        
        if avg_diff < 1e-6:
            status = "✅ 完美"
        elif avg_diff < 1e-5:
            status = "✅ 优秀"
        elif avg_diff < 1e-4:
            status = "✅ 良好"
        elif avg_diff < 1e-3:
            status = "⚠️  一般"
        else:
            status = "❌ 差"
        
        return f"{status} (MAE: {avg_diff:.6e}, Max: {max_diff:.6e}, Rel: {rel_error:.4%})"
    
    # 转换为字符串进行处理
    val1_str = str(val1)
    val2_str = str(val2)
    
    # 尝试解析为浮点数
    try:
        # 如果是多个值（用空格分隔的字符串）
        if " " in val1_str and " " in val2_str:
            vals1 = [float(v) for v in val1_str.split()]
            vals2 = [float(v) for v in val2_str.split()]
            
            if len(vals1) != len(vals2):
                return f"❌ 数量不同 ({len(vals1)} vs {len(vals2)})"
            
            diffs = [abs(v1 - v2) for v1, v2 in zip(vals1, vals2)]
            max_diff = max(diffs)
            avg_diff = sum(diffs) / len(diffs)
            
            # 相对误差
            avg_val = sum(abs(v) for v in vals1) / len(vals1)
            rel_error = avg_diff / (avg_val + 1e-10)
            
            if avg_diff < 1e-6:
                status = "✅ 完美"
            elif avg_diff < 1e-5:
                status = "✅ 优秀"
            elif avg_diff < 1e-4:
                status = "✅ 良好"
            elif avg_diff < 1e-3:
                status = "⚠️  一般"
            else:
                status = "❌ 差"
            
            return f"{status} (MAE: {avg_diff:.6e}, Max: {max_diff:.6e}, Rel: {rel_error:.4%})"
        
        # 单个数值
        else:
            val1_num = float(val1_str)
            val2_num = float(val2_str)
            diff = abs(val1_num - val2_num)
            rel_error = diff / (abs(val1_num) + 1e-10)
            
            if diff < 1e-6:
                status = "✅ 完美"
            elif diff < 1e-5:
                status = "✅ 优秀"
            elif diff < 1e-4:
                status = "✅ 良好"
            elif diff < 1e-3:
                status = "⚠️  一般"
            else:
                status = "❌ 差"
            
            return f"{status} (Diff: {diff:.6e}, Rel: {rel_error:.4%})"
    
    except (ValueError, ZeroDivisionError):
        # 非数值比较
        if val1_str == val2_str:
            return "✅ 相同"
        else:
            return f"❌ 不同"


def compare_dict_recursive(dict1, dict2, path=""):
    """递归对比两个字典"""
    comparisons = []
    
    all_keys = set(dict1.keys()) | set(dict2.keys())
    
    for key in sorted(all_keys):
        current_path = f"{path}.{key}" if path else key
        
        if key not in dict1:
            comparisons.append((current_path, None, dict2[key], "⚠️  仅在文件2中存在"))
            continue
        if key not in dict2:
            comparisons.append((current_path, dict1[key], None, "⚠️  仅在文件1中存在"))
            continue
        
        val1 = dict1[key]
        val2 = dict2[key]
        
        # 如果都是字典，递归对比
        if isinstance(val1, dict) and isinstance(val2, dict):
            comparisons.extend(compare_dict_recursive(val1, val2, current_path))
        else:
            # 否则直接对比值
            comparison = compare_values(val1, val2, key)
            comparisons.append((current_path, val1, val2, comparison))
    
    return comparisons


def compare_results(file1, file2, module_filter=None):
    """对比两个结果文件"""
    print(f"\n{'='*80}")
    print("精度对比")
    print('='*80)
    print(f"文件1: {file1}")
    print(f"文件2: {file2}")
    print('='*80)
    
    results1 = parse_results_file(file1)
    results2 = parse_results_file(file2)
    
    # 获取所有模块
    all_modules = set(results1.keys()) | set(results2.keys())
    
    if module_filter:
        if module_filter not in all_modules:
            print(f"\n❌ 模块 '{module_filter}' 不存在")
            print(f"可用模块: {', '.join(sorted(all_modules))}")
            return
        modules_to_compare = [module_filter]
    else:
        modules_to_compare = sorted(all_modules)
    
    print(f"\n模块数量: 文件1有{len(results1)}个, 文件2有{len(results2)}个")
    if module_filter:
        print(f"对比模块: {module_filter}")
    
    # 对比每个模块
    for module_name in modules_to_compare:
        print(f"\n{'='*80}")
        print(f"模块: {module_name}")
        print('='*80)
        
        if module_name not in results1:
            print("⚠️  仅在文件2中存在")
            continue
        if module_name not in results2:
            print("⚠️  仅在文件1中存在")
            continue
        
        module1 = results1[module_name]
        module2 = results2[module_name]
        
        # 递归对比模块中的所有数据
        comparisons = compare_dict_recursive(module1, module2)
        
        # 打印对比结果
        for path, val1, val2, comparison in comparisons:
            # 跳过描述性字段
            if any(skip in path for skip in ['description', 'module_name', 'input_source']):
                continue
            
            print(f"\n  {path}:")
            if val1 is not None:
                val1_str = json.dumps(val1, ensure_ascii=False) if isinstance(val1, (dict, list)) else str(val1)
                if len(val1_str) > 100:
                    val1_str = val1_str[:100] + "..."
                print(f"    文件1: {val1_str}")
            if val2 is not None:
                val2_str = json.dumps(val2, ensure_ascii=False) if isinstance(val2, (dict, list)) else str(val2)
                if len(val2_str) > 100:
                    val2_str = val2_str[:100] + "..."
                print(f"    文件2: {val2_str}")
            print(f"    对比: {comparison}")
    
    print("\n" + "="*80)
    print("对比完成")
    print("="*80)
    print("\n精度等级说明:")
    print("  ✅ 完美: Diff < 1e-6")
    print("  ✅ 优秀: Diff < 1e-5")
    print("  ✅ 良好: Diff < 1e-4")
    print("  ⚠️  一般: Diff < 1e-3")
    print("  ❌ 差:   Diff > 1e-3")


def main():
    parser = argparse.ArgumentParser(
        description="精度对比工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 对比两个文件
  python compare_precision.py pytorch_results.txt paddle_results.txt
  
  # 只对比特定模块
  python compare_precision.py pytorch_results.txt paddle_results.txt --module vae_encoder
        """
    )
    
    parser.add_argument('file1', help='第一个结果文件（如PyTorch）')
    parser.add_argument('file2', help='第二个结果文件（如Paddle）')
    parser.add_argument('--module', '-m', help='只对比指定的模块')
    
    args = parser.parse_args()
    
    file1 = Path(args.file1)
    file2 = Path(args.file2)
    
    if not file1.exists():
        print(f"❌ 文件不存在: {file1}")
        return 1
    
    if not file2.exists():
        print(f"❌ 文件不存在: {file2}")
        return 1
    
    try:
        compare_results(file1, file2, args.module)
        return 0
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
