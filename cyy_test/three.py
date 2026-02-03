"""
完整的 Chemeleon2 基础功能测试案例

测试流程：
1. 模型加载验证 - 验证 VAE 和 LDM 模型是否可以正确加载
2. 结构生成 - 使用预训练模型生成晶体结构
3. 结构验证 - 检查生成的结构是否有效
4. 指标计算 - 计算生成结构的各项评估指标
5. 结果分析 - 验证结果是否满足预期

预期结果：
- 能够成功生成指定数量的晶体结构
- 生成的结构都是有效的 Pymatgen Structure 对象
- 独特性(Uniqueness) > 80%
- 有效性(Validity) = 100%
- 能够正确保存和加载结果
"""

import os
import sys
from pathlib import Path

import numpy as np
import torch
from monty.serialization import dumpfn, loadfn
from pymatgen.core import Structure

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.sample import sample
from src.utils.metrics import Metrics


def test_environment():
    """测试 1: 验证运行环境"""
    print("=" * 80)
    print("测试 1: 验证运行环境")
    print("=" * 80)
    
    # 检查 CUDA 是否可用
    cuda_available = torch.cuda.is_available()
    print(f"✓ CUDA 可用: {cuda_available}")
    if cuda_available:
        print(f"  - GPU 设备数量: {torch.cuda.device_count()}")
        print(f"  - 当前设备: {torch.cuda.current_device()}")
        print(f"  - 设备名称: {torch.cuda.get_device_name(0)}")
    
    # 检查必要的目录是否存在
    required_dirs = [
        "checkpoints",
        "configs",
        "src",
        "outputs",
    ]
    
    for dir_name in required_dirs:
        dir_path = project_root / dir_name
        exists = dir_path.exists()
        print(f"✓ 目录 {dir_name} 存在: {exists}")
        assert exists, f"必需目录 {dir_name} 不存在"
    
    print("\n✅ 环境验证通过\n")
    return True


def test_model_sampling():
    """测试 2: 模型采样功能"""
    print("=" * 80)
    print("测试 2: 模型采样功能")
    print("=" * 80)
    
    # 设置参数
    num_samples = 50  # 测试用小批量
    batch_size = 25
    output_dir = "cyy_test/outputs/test_three"
    
    print(f"参数设置:")
    print(f"  - 样本数量: {num_samples}")
    print(f"  - 批次大小: {batch_size}")
    print(f"  - 输出目录: {output_dir}")
    
    try:
        # 执行采样
        print("\n开始生成晶体结构...")
        gen_atoms_list = sample(
            num_samples=num_samples,
            batch_size=batch_size,
            output_dir=output_dir,
            num_atom_distribution="mp-20",
            sampling_steps=20,  # 减少采样步数以加快测试
            save_json=True,
        )
        
        print(f"\n✓ 成功生成 {len(gen_atoms_list)} 个结构")
        
        # 验证生成的结构数量
        assert len(gen_atoms_list) == num_samples, \
            f"生成数量不匹配: 期望 {num_samples}, 实际 {len(gen_atoms_list)}"
        
        print("✅ 模型采样测试通过\n")
        return gen_atoms_list, output_dir
        
    except Exception as e:
        print(f"\n❌ 模型采样失败: {str(e)}")
        raise


def test_structure_validity(gen_atoms_list):
    """测试 3: 验证生成结构的有效性"""
    print("=" * 80)
    print("测试 3: 验证生成结构的有效性")
    print("=" * 80)
    
    valid_count = 0
    invalid_structures = []
    
    for idx, atoms in enumerate(gen_atoms_list):
        try:
            # 检查是否为有效的 ASE Atoms 对象
            assert hasattr(atoms, 'get_positions'), "不是有效的 ASE Atoms 对象"
            assert hasattr(atoms, 'get_chemical_symbols'), "缺少化学符号信息"
            assert hasattr(atoms, 'get_cell'), "缺少晶胞信息"
            
            # 检查原子数量
            num_atoms = len(atoms)
            assert num_atoms > 0, "结构中没有原子"
            
            # 检查晶胞体积
            volume = atoms.get_volume()
            assert volume > 0, "晶胞体积为负或零"
            
            # 尝试转换为 Pymatgen Structure
            structure = Structure(
                lattice=atoms.get_cell(),
                species=atoms.get_chemical_symbols(),
                coords=atoms.get_positions(),
                coords_are_cartesian=True,
            )
            
            # 验证 Pymatgen Structure 的基本属性
            assert structure.num_sites > 0, "Pymatgen 结构中没有原子"
            assert structure.composition is not None, "缺少组成信息"
            assert structure.volume > 0, "Pymatgen 结构体积无效"
            
            valid_count += 1
            
        except Exception as e:
            invalid_structures.append((idx, str(e)))
            print(f"  ⚠ 结构 {idx} 无效: {str(e)}")
    
    # 计算有效性指标
    validity = valid_count / len(gen_atoms_list) * 100
    print(f"\n有效结构统计:")
    print(f"  - 有效数量: {valid_count}/{len(gen_atoms_list)}")
    print(f"  - 有效率: {validity:.2f}%")
    
    # 显示一些基本统计
    if valid_count > 0:
        num_atoms_list = [len(atoms) for atoms in gen_atoms_list[:valid_count]]
        volumes_list = [atoms.get_volume() for atoms in gen_atoms_list[:valid_count]]
        
        print(f"\n结构统计:")
        print(f"  - 原子数量范围: {min(num_atoms_list)} - {max(num_atoms_list)}")
        print(f"  - 平均原子数: {np.mean(num_atoms_list):.2f}")
        print(f"  - 晶胞体积范围: {min(volumes_list):.2f} - {max(volumes_list):.2f} Å³")
        print(f"  - 平均晶胞体积: {np.mean(volumes_list):.2f} Å³")
    
    # 验证有效性是否达到预期
    assert validity == 100.0, f"结构有效性 {validity:.2f}% 未达到预期 100%"
    
    print("\n✅ 结构有效性测试通过\n")
    return valid_count


def test_metrics_calculation(output_dir):
    """测试 4: 计算和验证评估指标"""
    print("=" * 80)
    print("测试 4: 计算和验证评估指标")
    print("=" * 80)
    
    try:
        # 加载生成的结构
        structure_file = Path(output_dir) / "generated_structures.json.gz"
        assert structure_file.exists(), f"结构文件不存在: {structure_file}"
        
        gen_structures = loadfn(structure_file)
        print(f"✓ 成功加载 {len(gen_structures)} 个结构")
        
        # 初始化指标计算器
        print("\n初始化指标计算器...")
        metrics = Metrics(
            metrics=["unique", "novel", "e_above_hull"],
            reference_dataset="mp-20",
            phase_diagram="mp-all",
            metastable_threshold=0.1,
        )
        
        # 计算指标
        print("计算评估指标...")
        results = metrics.compute(gen_structures=gen_structures)
        
        # 提取关键指标
        uniqueness = results["unique"].mean() * 100
        novelty = results["novel"].mean() * 100
        
        print(f"\n评估指标:")
        print(f"  - 独特性 (Uniqueness): {uniqueness:.2f}%")
        print(f"  - 新颖性 (Novelty): {novelty:.2f}%")
        
        # 如果包含稳定性指标
        if "is_metastable" in results:
            metastable = results["is_metastable"].mean() * 100
            print(f"  - 亚稳态 (Metastable): {metastable:.2f}%")
            
            # 计算 mSUN 分数
            msun_score = (
                results["unique"] & results["novel"] & results["is_metastable"]
            ).mean() * 100
            print(f"  - mSUN 分数: {msun_score:.2f}%")
        
        # 如果包含能量指标
        if "e_above_hull" in results:
            e_above_hull_values = results["e_above_hull"]
            # 过滤掉 NaN 值
            valid_energies = e_above_hull_values[~np.isnan(e_above_hull_values)]
            if len(valid_energies) > 0:
                avg_energy = np.mean(valid_energies)
                print(f"  - 平均相对能量 (Avg E above hull): {avg_energy:.4f} eV/atom")
        
        # 保存结果
        df = metrics.to_dataframe()
        output_csv = Path(output_dir) / "metrics_results.csv"
        df.to_csv(output_csv, index=False)
        print(f"\n✓ 指标结果已保存至: {output_csv}")
        
        # 验证指标是否满足预期
        print(f"\n验证指标是否满足预期:")
        
        # 独特性应该大于 80%
        assert uniqueness > 80.0, f"独特性 {uniqueness:.2f}% 未达到预期 >80%"
        print(f"  ✓ 独特性达标 (>80%)")
        
        # 新颖性应该大于 0% (至少有一些新结构)
        assert novelty > 0.0, f"新颖性 {novelty:.2f}% 为 0，没有新结构"
        print(f"  ✓ 新颖性达标 (>0%)")
        
        print("\n✅ 指标计算测试通过\n")
        return results
        
    except Exception as e:
        print(f"\n❌ 指标计算失败: {str(e)}")
        raise


def test_file_io(output_dir):
    """测试 5: 文件读写功能"""
    print("=" * 80)
    print("测试 5: 文件读写功能")
    print("=" * 80)
    
    output_path = Path(output_dir)
    
    # 检查生成的文件
    expected_files = [
        "generated_structures.json.gz",
        "metrics_results.csv",
    ]
    
    print("检查生成的文件:")
    for filename in expected_files:
        file_path = output_path / filename
        exists = file_path.exists()
        if exists:
            size = file_path.stat().st_size
            print(f"  ✓ {filename} (大小: {size:,} bytes)")
        else:
            print(f"  ✗ {filename} 不存在")
        assert exists, f"期望的文件不存在: {filename}"
    
    # 测试结构文件的读写
    print("\n测试结构文件的读写:")
    structure_file = output_path / "generated_structures.json.gz"
    
    # 读取
    structures = loadfn(structure_file)
    print(f"  ✓ 成功读取 {len(structures)} 个结构")
    
    # 验证结构类型
    assert all(isinstance(s, Structure) for s in structures), \
        "不是所有对象都是 Pymatgen Structure"
    print(f"  ✓ 所有对象都是有效的 Pymatgen Structure")
    
    # 测试重新保存
    test_file = output_path / "test_save.json.gz"
    dumpfn(structures[:5], test_file)  # 只保存前5个结构测试
    print(f"  ✓ 成功保存测试文件")
    
    # 测试重新读取
    test_structures = loadfn(test_file)
    assert len(test_structures) == 5, "读取的结构数量不匹配"
    print(f"  ✓ 成功读取测试文件")
    
    # 清理测试文件
    test_file.unlink()
    print(f"  ✓ 清理测试文件")
    
    print("\n✅ 文件读写测试通过\n")
    return True


def run_all_tests():
    """运行所有测试"""
    print("\n" + "=" * 80)
    print(" " * 20 + "Chemeleon2 完整功能测试")
    print("=" * 80 + "\n")
    
    test_results = {}
    
    try:
        # 测试 1: 环境验证
        test_results["environment"] = test_environment()
        
        # 测试 2: 模型采样
        gen_atoms_list, output_dir = test_model_sampling()
        test_results["sampling"] = True
        
        # 测试 3: 结构有效性
        valid_count = test_structure_validity(gen_atoms_list)
        test_results["validity"] = True
        
        # 测试 4: 指标计算
        results = test_metrics_calculation(output_dir)
        test_results["metrics"] = True
        
        # 测试 5: 文件读写
        test_results["file_io"] = test_file_io(output_dir)
        
    except Exception as e:
        print(f"\n❌ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    # 打印测试总结
    print("\n" + "=" * 80)
    print(" " * 30 + "测试总结")
    print("=" * 80)
    
    all_passed = all(test_results.values())
    
    for test_name, passed in test_results.items():
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"  {test_name.ljust(20)}: {status}")
    
    print("=" * 80)
    
    if all_passed:
        print("\n🎉 所有测试通过！Chemeleon2 基础功能正常！\n")
        return True
    else:
        print("\n⚠️  部分测试失败，请检查错误信息\n")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
