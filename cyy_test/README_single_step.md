# 单步精度测试指南

## 概述

`test_single_step.py` 提供单模块测试和精度对比功能，支持固定输入确保可复现性。

## 核心特性

### 1. 固定输入测试

通过保存和重用输入数据，确保每次测试使用完全相同的输入：

```bash
# 第一次运行会自动保存输入数据
python cyy_test/test_single_step.py --module vae_encoder

# 后续使用固定输入
python cyy_test/test_single_step.py --module vae_encoder --use-fixed-input
```

**验证结果**：两次使用固定输入的测试，所有输出的 MAE = 0.000000e+00 ✅

### 2. 单模块测试

只测试特定模块，节省时间：

```bash
# 测试 VAE Encoder
python cyy_test/test_single_step.py --module vae_encoder

# 测试 LDM Denoiser
python cyy_test/test_single_step.py --module ldm_denoiser --use-fixed-input
```

支持的模块：
- `vae_encoder` - VAE 编码器
- `vae_decoder` - VAE 解码器
- `vae_full` - 完整 VAE (编码+解码)
- `ldm_denoiser` - LDM 去噪器
- `ldm_sampling` - LDM 采样

### 3. 精度对比

对比两个框架（如 PyTorch vs Paddle）的输出精度：

```bash
# 对比两次测试结果
python cyy_test/test_single_step.py --module vae_encoder --compare \
    --output pytorch_results.npz \
    --compare-with paddle_results.npz
```

**对比指标**：
- MAE (Mean Absolute Error) - 平均绝对误差
- MSE (Mean Square Error) - 均方误差  
- Max Diff - 最大差异
- Relative Error - 相对误差

**精度等级**：
- ✅ 优秀: MAE < 1e-5
- ✅ 良好: MAE < 1e-4
- ⚠️  一般: MAE < 1e-3
- ❌ 差: MAE > 1e-3

## 完整工作流程

### PyTorch 侧

```bash
# 1. 首次运行，生成基准结果和输入数据
python cyy_test/test_single_step.py --module vae_encoder \
    --output cyy_test/outputs/precision_test/pytorch_vae_encoder.npz

# 2. 验证可复现性（使用固定输入再测一次）
python cyy_test/test_single_step.py --module vae_encoder \
    --use-fixed-input \
    --fixed-input-path cyy_test/outputs/precision_test/pytorch_vae_encoder.npz \
    --output cyy_test/outputs/precision_test/pytorch_vae_encoder_v2.npz

# 3. 对比两次结果（应该完全一致）
python cyy_test/test_single_step.py --module vae_encoder --compare \
    --output cyy_test/outputs/precision_test/pytorch_vae_encoder.npz \
    --compare-with cyy_test/outputs/precision_test/pytorch_vae_encoder_v2.npz
```

### Paddle 侧（迁移后）

```bash
# 1. 加载 PyTorch 的固定输入
# 在 Paddle 脚本中读取：
# fixed_inputs = np.load('pytorch_vae_encoder.npz', allow_pickle=True)
# input_data = fixed_inputs['vae_encoder'].item()['input']

# 2. 使用相同输入测试 Paddle 模型
python paddle_test_single_step.py --module vae_encoder \
    --use-fixed-input \
    --fixed-input-path cyy_test/outputs/precision_test/pytorch_vae_encoder.npz \
    --output cyy_test/outputs/precision_test/paddle_vae_encoder.npz

# 3. 对比 PyTorch vs Paddle
python cyy_test/test_single_step.py --module vae_encoder --compare \
    --output cyy_test/outputs/precision_test/pytorch_vae_encoder.npz \
    --compare-with cyy_test/outputs/precision_test/paddle_vae_encoder.npz
```

## 命令行参数

```bash
python cyy_test/test_single_step.py [OPTIONS]

Options:
  --module {vae_encoder,vae_decoder,vae_full,ldm_denoiser,ldm_sampling}
                        要测试的模块 (默认: vae_encoder)
  
  --use-fixed-input     使用固定输入（从之前保存的结果加载）
  
  --fixed-input-path PATH
                        固定输入数据的路径
                        (默认: cyy_test/outputs/precision_test/pytorch_results.npz)
  
  --output PATH         输出文件路径
                        (默认: cyy_test/outputs/precision_test/single_step_results.npz)
  
  --compare             对比两次测试结果
  
  --compare-with PATH   要对比的另一个结果文件
                        (默认: cyy_test/outputs/precision_test/pytorch_results.npz)
  
  --seed INT            随机种子 (默认: 42)
  
  --device {cuda,cpu}   设备 (默认: cuda)
```

## 示例输出

### 测试输出

```
================================================================================
单步测试: vae_encoder
================================================================================

使用固定输入数据

Encoder Output (x):
  Shape: (12, 512)
  Mean: 0.018948
  Std: 0.948124
  Min: -5.150079
  Max: 15.993896
  First 4 values: [-0.179900 -0.046454 -0.293423 -0.057796]

Latent Mean:
  Shape: (12, 8)
  Mean: 0.231387
  Std: 1.060186
  First 4 values: [1.809650 0.524041 -0.375436 -1.061384]

✓ 结果已保存至: cyy_test/outputs/precision_test/test1.npz
✓ 日志已保存至: cyy_test/outputs/precision_test/single_step_vae_encoder.log
```

### 对比输出

```
================================================================================
精度对比: vae_encoder
================================================================================

对比 pytorch_results.npz vs paddle_results.npz:

  encoder_output:
    MAE: 1.234567e-05  MSE: 2.345678e-10  Max: 5.678901e-05
    相对误差: 0.01%  ✅ 优秀
    
  latent_mean:
    MAE: 3.456789e-06  MSE: 1.234567e-11  Max: 2.345678e-05
    相对误差: 0.003%  ✅ 优秀
    
  latent_z:
    MAE: 8.901234e-05  MSE: 7.890123e-09  Max: 4.567890e-04
    相对误差: 0.08%  ✅ 良好

精度等级说明:
  ✅ 优秀: MAE < 1e-5
  ✅ 良好: MAE < 1e-4
  ⚠️  一般: MAE < 1e-3
  ❌ 差:   MAE > 1e-3
```

## 保存的数据格式

每个测试结果 npz 文件包含：

```python
{
    "vae_encoder": {
        "input": {  # 输入数据（用于固定输入）
            "atom_types": array(...),
            "frac_coords": array(...),
            "lattices": array(...),
            "num_atoms": array(...),
            ...
        },
        "encoder_output": array(...),     # 编码器输出
        "quant_conv_output": array(...),  # 量化卷积输出
        "latent_mean": array(...),        # 潜在均值
        "latent_logvar": array(...),      # 潜在对数方差
        "latent_z": array(...),           # 采样的潜在向量
        "random_eps": array(...),         # 随机噪声（用于复现采样）
    }
}
```

## 加载固定输入示例（Python）

```python
import numpy as np
import torch

# 1. 加载 PyTorch 的输入数据
pytorch_data = np.load('pytorch_vae_encoder.npz', allow_pickle=True)
input_data = pytorch_data['vae_encoder'].item()['input']

# 2. 在 Paddle 中使用相同输入
import paddle

atom_types = paddle.to_tensor(input_data['atom_types'], dtype='int64')
frac_coords = paddle.to_tensor(input_data['frac_coords'], dtype='float32')
lattices = paddle.to_tensor(input_data['lattices'], dtype='float32')

# 3. 运行 Paddle 模型
paddle_output = paddle_model(atom_types, frac_coords, lattices)

# 4. 保存 Paddle 结果
np.savez('paddle_vae_encoder.npz', 
         vae_encoder={
             'input': input_data,  # 保存相同的输入
             'encoder_output': paddle_output.numpy(),
             ...
         })

# 5. 使用 test_single_step.py 对比精度
# python cyy_test/test_single_step.py --module vae_encoder --compare \
#     --output pytorch_vae_encoder.npz --compare-with paddle_vae_encoder.npz
```

## 优势

1. **确定性**：固定随机种子 + 固定输入 = 完全可复现
2. **高效**：只测试需要的模块，节省时间
3. **精确**：逐层对比，快速定位精度问题
4. **便捷**：自动保存/加载输入，无需手动管理

## 常见问题

### Q1: 为什么使用固定输入？

A: 框架迁移时，输入必须完全相同才能准确对比输出精度。固定输入消除了随机性。

### Q2: 随机数如何固定？

A: 通过 `--seed` 参数设置随机种子（默认42），确保每次生成相同的随机数。

### Q3: 如何确保输入完全一致？

A: 使用 `--use-fixed-input` 从 npz 文件加载之前保存的输入数据。

### Q4: 对比结果显示 MAE > 1e-3，怎么办？

A: 
1. 检查模型权重是否正确转换
2. 检查前向传播逻辑是否一致
3. 检查数据类型和精度（float32 vs float64）
4. 逐层对比，定位问题层

### Q5: 如何测试 LDM Denoiser？

A: LDM Denoiser 也支持固定输入，且会自动保存输入的 z 和 t：

```bash
python cyy_test/test_single_step.py --module ldm_denoiser --use-fixed-input
```

## 相关文件

- `test_single_step.py` - 单步测试主脚本
- `test_precision.py` - 完整精度测试脚本
- `README_precision.md` - 完整精度测试文档
- `view_precision_log.sh` - 日志查看工具
