# 完整Tensor数据导出说明

## 概述

`test_precision.py` 现在支持导出完整的tensor数据到NPZ文件，同时保留JSON摘要格式。

## 为什么需要完整数据？

### 之前的问题
- JSON只包含`first_4_values`，缺少完整tensor数据
- 无法进行精确的数值对比
- 跨框架验证困难

### 现在的解决方案
- **JSON文件**: 包含结构化摘要 + NPZ文件路径
- **NPZ文件**: 保存完整的tensor数据（每个tensor一个独立文件）

## 输出文件结构

```
cyy_test/outputs/precision_test/
├── pytorch_results.json                      # JSON摘要（推荐AI工具先读这个）
├── pytorch_results_readable.txt              # 易读文本版本
├── precision_test_log.txt                    # 详细日志
│
├── vae_encoder_input_atom_types_full.npz    # VAE编码器输入数据
├── vae_encoder_input_frac_coords_full.npz
├── vae_encoder_input_lattices_full.npz
├── vae_encoder_output_full.npz              # VAE编码器输出数据
├── vae_encoder_latent_mean_full.npz
├── vae_encoder_latent_z_full.npz
│
├── vae_decoder_input_z_full.npz             # VAE解码器数据
├── vae_decoder_atom_types_full.npz
│
├── ldm_denoiser_input_z_full.npz            # LDM去噪器数据
├── ldm_denoiser_output_noise_full.npz
│
└── ldm_sampling_coords_full.npz             # LDM采样数据
    └── ldm_sampling_lattices_full.npz
```

## 快速开始

### 1. 运行测试并生成数据

```bash
cd /home/cao/opensource/cailiao/chemeleon2
python cyy_test/test_precision.py
```

### 2. 查看JSON摘要（推荐AI工具使用）

```python
import json

with open('cyy_test/outputs/precision_test/pytorch_results.json') as f:
    results = json.load(f)

# 查看VAE编码器的输出摘要
encoder_output = results['modules']['vae_encoder']['output']['latent_mean']
print(f"形状: {encoder_output['shape']}")
print(f"前4个值: {encoder_output['first_4_values']}")
print(f"统计: mean={encoder_output['mean']}, std={encoder_output['std']}")
print(f"完整数据路径: {encoder_output['full_data_file']}")
```

### 3. 加载完整NPZ数据（用于精确对比）

```python
import numpy as np

# 从JSON获取NPZ文件路径
npz_path = encoder_output['full_data_file']

# 加载完整数据
data = np.load(npz_path)
tensor = data['data']  # NPZ中的数据存储在'data'键

print(f"完整形状: {tensor.shape}")
print(f"完整统计: mean={tensor.mean()}, std={tensor.std()}")
print(f"前10个值: {tensor.flatten()[:10]}")
```

### 4. 对比两个框架的结果

```python
# 加载PyTorch结果
pytorch_tensor = np.load('pytorch_vae_encoder_latent_mean_full.npz')['data']

# 加载Paddle结果（假设你已经用Paddle运行了相同的测试）
paddle_tensor = np.load('paddle_vae_encoder_latent_mean_full.npz')['data']

# 计算差异
diff = np.abs(pytorch_tensor - paddle_tensor)
print(f"绝对误差: max={diff.max():.2e}, mean={diff.mean():.2e}")
print(f"相对误差: {(diff / (np.abs(pytorch_tensor) + 1e-8)).max():.2e}")
```

## 工作流程建议

### 对于AI工具：

1. **首先读取JSON文件** (`pytorch_results.json`)
   - 获取结构化的测试信息
   - 查看每个模块的输入输出描述
   - 了解数据的形状、统计量、前4个值

2. **当需要完整数据时，加载NPZ文件**
   - 从JSON中获取`full_data_file`路径
   - 使用`np.load(path)['data']`加载完整tensor
   - 进行精确的数值对比

3. **对比不同框架的结果**
   - 使用相同的输入数据（可以从NPZ加载）
   - 比较输出tensor的差异
   - 生成详细的对比报告

## 示例脚本

我们提供了完整的示例脚本：

```bash
# 查看如何使用NPZ文件
python cyy_test/load_npz_example.py
```

这个脚本包含三个示例：
1. 基本的NPZ文件加载
2. 对比两个框架的结果
3. 遍历所有模块的输出

## JSON格式说明

每个tensor在JSON中的结构：

```json
{
  "shape": [12, 8],
  "dtype": "torch.float32",
  "device": "cuda:0",
  "first_4_values": [1.809, 0.524, -0.375, -1.061],
  "total_elements": 96,
  "mean": 0.231387,
  "std": 1.060186,
  "min": -2.098549,
  "max": 2.806265,
  "full_data_file": "cyy_test/outputs/precision_test/vae_encoder_latent_mean_full.npz",
  "full_data_available": true
}
```

## NPZ格式说明

每个NPZ文件：
- 是压缩的NumPy数组存档格式
- 包含一个键`'data'`，值是完整的tensor数组
- 可以用任何支持NumPy的工具读取
- 保持原始数值精度（float32/float64）

## 相关文档

- [README_JSON_format.md](README_JSON_format.md) - JSON格式详细说明
- [README_npz.md](README_npz.md) - NPZ格式和使用指南
- [load_npz_example.py](load_npz_example.py) - 完整的使用示例

## 常见问题

### Q: JSON和NPZ数据一致吗？
A: 是的。JSON中的`first_4_values`、`mean`、`std`等统计量都是从NPZ中的完整数据计算得出的。

### Q: 为什么不把完整数据放在JSON里？
A: 大型tensor会让JSON文件过大且难以阅读。分离的设计让AI工具可以快速浏览JSON，只在需要时加载NPZ。

### Q: 如何确保两个框架使用相同的输入？
A: 可以从PyTorch的NPZ文件中加载输入数据，然后在Paddle中使用相同的数据。这样确保输入完全一致。

### Q: NPZ文件可以跨平台使用吗？
A: 是的。NPZ是NumPy的标准格式，可以在任何支持NumPy的平台上加载。

## 技术细节

### 生成流程

1. 运行测试（PyTorch前向传播）
2. 对于每个tensor：
   - 计算统计量（mean, std, min, max）
   - 提取前4个值
   - 保存完整数据到NPZ文件
   - 在JSON中记录摘要和NPZ路径
3. 组装完整的JSON结构
4. 保存JSON和易读文本

### 存储效率

- NPZ使用压缩（`np.savez_compressed`）
- 典型的8x12 float32 tensor: ~600 bytes
- 大型512x12 float32 tensor: ~23 KB
- 总存储（所有NPZ文件）: 通常 < 1 MB

## 更新日志

### 2024-02-02
- ✅ 添加NPZ文件导出功能
- ✅ 在JSON中添加`full_data_file`和`full_data_available`字段
- ✅ 创建`load_npz_example.py`示例脚本
- ✅ 修复所有测试函数的return语句
- ✅ 更新文档说明
