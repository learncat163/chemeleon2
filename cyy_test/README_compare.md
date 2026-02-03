# 精度对比使用指南

## 概述

用于PyTorch和Paddle框架之间的精度对比工具。输出格式为简单的TXT文本，易于阅读和对比。

## 输出格式说明

测试结果保存为文本文件（.txt），格式为键值对，每个张量只保留前4个值（如果维度>4）。

### 文件结构

```
================================================================================
Chemeleon2 精度测试结果 - PyTorch
================================================================================
测试时间: 2026-02-02 15:11:04
随机种子: 42
设备: cuda
================================================================================

================================================================================
模块: vae_encoder
================================================================================

encoder_output_dtype: torch.float32
encoder_output_first_4_values: -0.17990004 -0.04645369 -0.29342285 -0.05779608
encoder_output_max: 15.99389553
encoder_output_mean: 0.01894787
encoder_output_min: -5.15007877
encoder_output_shape: (12, 512)
encoder_output_std: 0.94812447
...
```

### 数据格式

- **shape**: 张量形状，如 `(12, 512)`
- **dtype**: 数据类型，如 `torch.float32`
- **mean**: 均值，8位小数，如 `0.01894787`
- **std**: 标准差，8位小数
- **min**: 最小值，8位小数
- **max**: 最大值，8位小数
- **first_4_values**: 前4个值（空格分隔），8位小数，如 `-0.17990004 -0.04645369 -0.29342285 -0.05779608`

## 测试模块

测试脚本会测试以下5个模块：

1. **vae_encoder** - VAE编码器
   - encoder_output: 编码器原始输出
   - quant_conv_output: 量化卷积后的输出
   - latent_mean: 潜在向量均值
   - latent_logvar: 潜在向量对数方差
   - latent_z: 采样后的潜在向量

2. **vae_decoder** - VAE解码器
   - decoder_input_z: 解码器输入
   - decoder_post_quant: 后量化结果
   - decoder_output_atom_types: 重建的原子类型
   - decoder_output_frac_coords: 重建的分数坐标
   - decoder_output_lengths: 重建的晶格长度
   - decoder_output_angles: 重建的晶格角度

3. **vae_full** - 完整VAE测试
   - latent_mean: 编码得到的均值
   - latent_logvar: 编码得到的对数方差
   - coord_mae: 坐标重建误差
   - lattice_mae: 晶格重建误差

4. **ldm_denoiser** - LDM去噪器
   - denoiser_input_z: 输入潜在向量
   - denoiser_input_t: 时间步
   - denoiser_output: 预测的噪声

5. **ldm_sampling** - LDM完整采样
   - generated_coords: 生成的坐标
   - generated_lattices: 生成的晶格

## 使用方法

### 1. 运行测试

```bash
# 在项目根目录运行
cd /home/cao/opensource/cailiao/chemeleon2

# PyTorch测试
.venv/bin/python cyy_test/test_precision.py

# Paddle测试（在Paddle环境中运行类似的脚本）
# paddle_python cyy_test/test_precision_paddle.py
```

### 2. 对比结果

#### 方法1: 使用diff命令（快速）

```bash
diff cyy_test/outputs/precision_test/pytorch_results.txt paddle_results.txt
```

#### 方法2: 使用对比工具（推荐）

```bash
# 对比所有模块
.venv/bin/python cyy_test/compare_precision.py \
    cyy_test/outputs/precision_test/pytorch_results.txt \
    paddle_results.txt

# 只对比特定模块
.venv/bin/python cyy_test/compare_precision.py \
    cyy_test/outputs/precision_test/pytorch_results.txt \
    paddle_results.txt \
    --module vae_encoder
```

### 3. 对比结果解读

对比工具会自动计算差异并给出等级评估：

- ✅ **完美** (Diff < 1e-6): 精度极高，几乎完全一致
- ✅ **优秀** (Diff < 1e-5): 精度很好，符合预期
- ✅ **良好** (Diff < 1e-4): 精度可接受
- ⚠️  **一般** (Diff < 1e-3): 精度较低，需要检查
- ❌ **差** (Diff > 1e-3): 精度不足，需要调试

对比输出示例：

```
================================================================================
模块: ldm_denoiser
================================================================================

  denoiser_input_z_first_4_values:
    文件1: 0.13913755 -0.10821664 -0.71742225 0.75664860
    文件2: 0.13913756 -0.10821664 -0.71742225 0.75664860
    对比: ✅ 完美 (MAE: 2.500000e-09, Max: 1.000000e-08, Rel: 0.0000%)

  denoiser_output_mean:
    文件1: 0.10624263
    文件2: 0.10624263
    对比: ✅ 完美 (Diff: 0.000000e+00, Rel: 0.0000%)
```

## 输出文件位置

- **PyTorch结果**: `cyy_test/outputs/precision_test/pytorch_results.txt`
- **详细日志**: `cyy_test/outputs/precision_test/precision_test_log.txt`
- **Paddle结果**: 需要在Paddle环境中运行测试脚本生成

## 注意事项

1. **随机种子**: 两个框架都必须使用相同的随机种子（42）
2. **输入数据**: 必须使用相同的测试结构文件
3. **模型权重**: 确保加载的是对应的检查点文件
4. **设备**: 建议都在GPU上运行以获得一致性
5. **数据类型**: 确保两个框架都使用float32
6. **前4个值**: 只保存前4个值用于快速对比，完整精度由统计值（mean/std/min/max）提供

## 常见问题

### Q1: 为什么只保留前4个值？

A: 前4个值用于快速验证数据是否一致，完整的统计信息（mean/std/min/max）已经能够准确反映整体精度差异。

### Q2: 如何查看更多维度的值？

A: 修改 `test_precision.py` 中的 `max_dims` 参数（默认为4）。

### Q3: 精度差异在什么范围内是可接受的？

A: 
- 浮点运算：< 1e-6 是理想的
- 复杂模型：< 1e-4 通常是可接受的
- 如果差异 > 1e-3，需要检查模型实现是否正确

### Q4: 如何调试精度问题？

A:
1. 首先对比 VAE Encoder 的输出
2. 如果编码器正确，再检查解码器
3. 最后检查 LDM 模块
4. 使用 `--module` 参数逐个模块对比

## 文件清单

- `test_precision.py` - 精度测试主脚本
- `compare_precision.py` - 精度对比工具
- `README_compare.md` - 本文档
- `outputs/precision_test/pytorch_results.txt` - 测试结果
- `outputs/precision_test/precision_test_log.txt` - 详细日志

## 示例工作流

```bash
# 1. 运行PyTorch测试
cd /home/cao/opensource/cailiao/chemeleon2
.venv/bin/python cyy_test/test_precision.py

# 2. 运行Paddle测试（示例）
# paddle_python cyy_test/test_precision_paddle.py

# 3. 对比结果
.venv/bin/python cyy_test/compare_precision.py \
    cyy_test/outputs/precision_test/pytorch_results.txt \
    cyy_test/outputs/precision_test/paddle_results.txt

# 4. 如果发现差异，逐个模块对比
.venv/bin/python cyy_test/compare_precision.py \
    cyy_test/outputs/precision_test/pytorch_results.txt \
    cyy_test/outputs/precision_test/paddle_results.txt \
    --module vae_encoder
```
