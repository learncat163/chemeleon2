# NPZ 文件查看指南

## NPZ 格式简介

**NPZ** = **N**um**P**y **Z**ipped Archive（NumPy 压缩存档）

- 本质是 **ZIP 格式**，可以用解压工具打开
- 包含多个 `.npy` 文件（NumPy 数组）
- 每个数组有独立的键名
- 支持压缩和未压缩两种模式

### 为什么使用 NPZ？

1. ✅ **跨平台**：Python、MATLAB、C++ 都能读取
2. ✅ **高效**：二进制格式，保存/加载速度快
3. ✅ **精确**：保持原始数值精度（float32/float64）
4. ✅ **结构化**：可保存多个数组，类似字典
5. ✅ **压缩**：可选压缩减小文件大小

## 新版本说明 (2024-02)

**test_precision.py** 现在生成两种格式的输出：

1. **JSON 文件** (`pytorch_results.json`)：
   - 包含结构化的测试结果
   - 每个tensor只保存前4个值和统计信息
   - 包含NPZ文件的路径引用
   - 方便AI工具读取和理解

2. **NPZ 文件** (单独的 `*_full.npz` 文件)：
   - 保存完整的tensor数据
   - 用于精确的数值对比
   - 支持跨框架验证（如PyTorch vs Paddle）

### 文件结构示例

```
cyy_test/outputs/precision_test/
├── pytorch_results.json                      # JSON摘要（含first_4_values）
├── pytorch_results_readable.txt              # 易读文本
├── precision_test_log.txt                    # 详细日志
├── vae_encoder_input_atom_types_full.npz    # 完整输入数据
├── vae_encoder_input_frac_coords_full.npz
├── vae_encoder_output_full.npz              # 完整输出数据
├── vae_encoder_latent_mean_full.npz
├── vae_encoder_latent_z_full.npz
└── ... (更多NPZ文件)
```

## 使用方法

### 1. 查看JSON摘要（推荐用于快速了解）

```bash
# 查看JSON文件
cat cyy_test/outputs/precision_test/pytorch_results.json | jq .

# 或使用Python
python -c "
import json
with open('cyy_test/outputs/precision_test/pytorch_results.json') as f:
    results = json.load(f)
    # 查看VAE编码器的输出摘要
    encoder = results['modules']['vae_encoder']
    print(encoder['output']['latent_mean'])
"
```

### 2. 加载完整NPZ数据（用于精确对比）

```python
import numpy as np

# 加载单个NPZ文件
npz_file = 'cyy_test/outputs/precision_test/vae_encoder_latent_mean_full.npz'
data = np.load(npz_file)
tensor = data['data']  # NPZ文件中的键固定为'data'

print(f'形状: {tensor.shape}')
print(f'数据类型: {tensor.dtype}')
print(f'统计: mean={tensor.mean():.6f}, std={tensor.std():.6f}')
print(f'前4个值: {tensor.flatten()[:4]}')

# 对比两个框架的结果
pytorch_data = np.load('pytorch_vae_encoder_latent_mean_full.npz')['data']
paddle_data = np.load('paddle_vae_encoder_latent_mean_full.npz')['data']

diff = np.abs(pytorch_data - paddle_data)
print(f'绝对误差: max={diff.max():.2e}, mean={diff.mean():.2e}')
print(f'相对误差: {np.max(diff / (np.abs(pytorch_data) + 1e-8)):.2e}')
```

### 3. 验证JSON和NPZ的一致性

```python
import numpy as np
import json

# 加载JSON摘要
with open('cyy_test/outputs/precision_test/pytorch_results.json') as f:
    results = json.load(f)
    encoder_output = results['modules']['vae_encoder']['output']['latent_mean']
    json_first_4 = encoder_output['first_4_values']
    npz_path = encoder_output['full_data_path']  # NPZ文件路径

# 加载NPZ完整数据
npz_data = np.load(npz_path)['data']
npz_first_4 = npz_data.flatten()[:4].tolist()

# 验证一致性
print(f'JSON first_4: {json_first_4}')
print(f'NPZ first_4: {npz_first_4}')
print(f'一致: {np.allclose(json_first_4, npz_first_4)}')
```

### 4. 详细查看（旧版本，需要view_npz.py）

```bash
# 显示具体数值（默认前20个）
python cyy_test/view_npz.py results.npz --detail

# 显示前10个值
python cyy_test/view_npz.py results.npz --key vae_encoder --detail --max-values 10

# 显示前50个值
python cyy_test/view_npz.py results.npz --detail -m 50
```

**输出示例**：
```
  latent_mean:
    类型: float32
    形状: (12, 8)
    统计: ...
    前 10 个值:
      [   0]: 1.809650e+00  5.240414e-01  -3.754361e-01  -1.061384e+00  -7.844588e-01
      [   5]: 1.094122e+00  2.782036e-01  9.792569e-01  6.211126e-01  -4.361534e-01
```

### 3. 对比两个文件

```bash
# 对比 PyTorch vs Paddle
python cyy_test/view_npz.py pytorch.npz --compare paddle.npz

# 只对比特定模块
python cyy_test/view_npz.py pytorch.npz -c paddle.npz --key vae_encoder
```

**输出示例**：
```
对比模块: vae_encoder

  encoder_output:
    MAE:      1.234567e-05
    Max Diff: 5.678901e-05
    相对误差: 0.01%
    状态:     ✅ 优秀

  latent_mean:
    MAE:      8.901234e-05
    Max Diff: 4.567890e-04
    相对误差: 0.08%
    状态:     ✅ 良好
```

### 4. 导出为文本

```bash
# 导出整个文件
python cyy_test/view_npz.py results.npz --export output.txt

# 只导出特定模块
python cyy_test/view_npz.py results.npz --key vae_encoder --export vae_encoder.txt
```

文本文件包含完整的数值数据，可用 Excel、文本编辑器打开。

## 其他查看方式

### 方式1: Python 交互式

```python
import numpy as np

# 加载文件
data = np.load('pytorch_results.npz', allow_pickle=True)

# 查看所有键
print(list(data.keys()))
# ['vae_encoder', 'vae_decoder', ...]

# 读取特定模块
vae_encoder = data['vae_encoder'].item()  # .item() 用于object类型

# 查看子键
print(vae_encoder.keys())
# dict_keys(['encoder_output', 'latent_mean', ...])

# 获取数据
encoder_output = vae_encoder['encoder_output']
print(encoder_output.shape)  # (12, 512)
print(encoder_output[:5, :5])  # 显示前5x5

# 计算统计
print(f"均值: {encoder_output.mean()}")
print(f"标准差: {encoder_output.std()}")
```

### 方式2: 解压查看

NPZ 本质是 ZIP 文件，可以直接解压：

```bash
# 复制为 .zip
cp pytorch_results.npz pytorch_results.zip

# 解压
unzip pytorch_results.zip

# 会得到多个 .npy 文件
ls *.npy
```

但解压后的 `.npy` 文件是二进制，需要 NumPy 才能读取。

### 方式3: IPython/Jupyter

```python
%load_ext autoreload
%autoreload 2

import numpy as np

data = np.load('pytorch_results.npz', allow_pickle=True)

# IPython 可以直接显示
data['vae_encoder'].item()['latent_mean']
```

### 方式4: MATLAB

```matlab
% MATLAB 可以读取 .npy 文件
data = readNPY('encoder_output.npy');
size(data)
mean(data(:))
```

## 命令行参数速查

```bash
python cyy_test/view_npz.py <文件> [选项]

选项:
  --key, -k <模块名>        只查看指定模块
  --detail, -d             显示详细数值
  --max-values, -m <数量>  显示的最大数值数量（默认20）
  --compare, -c <文件>     对比另一个NPZ文件
  --export, -e <输出文件>  导出为文本文件
  --help, -h               显示帮助
```

## 实际应用示例

### 示例1: 快速检查文件内容

```bash
# 看看文件里有什么
python cyy_test/view_npz.py pytorch_results.npz
```

### 示例2: 验证 VAE Encoder 输出

```bash
# 详细查看 encoder 输出
python cyy_test/view_npz.py pytorch_results.npz \
    --key vae_encoder --detail --max-values 50
```

### 示例3: 对比 PyTorch vs Paddle

```bash
# 运行 PyTorch 测试
python test_single_step.py --module vae_encoder --output pytorch.npz

# 运行 Paddle 测试（假设）
python paddle_test.py --module vae_encoder --output paddle.npz

# 对比精度
python view_npz.py pytorch.npz --compare paddle.npz --key vae_encoder
```

### 示例4: 导出给同事查看

```bash
# 导出为文本，可用 Excel 打开
python view_npz.py results.npz --export results.txt

# 只导出某个模块
python view_npz.py results.npz --key ldm_denoiser --export ldm.txt
```

## 文件格式详解

### 内部结构

```
pytorch_results.npz (ZIP压缩包)
├── vae_encoder.npy          # vae_encoder 模块数据（object类型，包含dict）
├── vae_decoder.npy          # vae_decoder 模块数据
├── vae_full.npy             # vae_full 模块数据
├── ldm_denoiser.npy         # ldm_denoiser 模块数据
└── ldm_sampling.npy         # ldm_sampling 模块数据
```

每个 `.npy` 文件包含一个 Python 字典（需要 `allow_pickle=True`）：

```python
{
    'encoder_output': np.array([...]),    # shape: (12, 512)
    'latent_mean': np.array([...]),       # shape: (12, 8)
    'latent_logvar': np.array([...]),     # shape: (12, 8)
    'latent_z': np.array([...]),          # shape: (12, 8)
    ...
}
```

### 数据类型

- `float32` - 单精度浮点（4字节/元素）
- `float64` - 双精度浮点（8字节/元素）
- `int64` - 64位整数（8字节/元素）
- `bool` - 布尔值（1字节/元素）

## 常见问题

### Q1: 如何提取特定数组到新文件？

```python
import numpy as np

# 加载
data = np.load('pytorch_results.npz', allow_pickle=True)

# 提取
encoder_output = data['vae_encoder'].item()['encoder_output']

# 保存为单独的 npy
np.save('encoder_output.npy', encoder_output)

# 或保存为新的 npz
np.savez('encoder_only.npz', encoder_output=encoder_output)
```

### Q2: 如何在 Paddle 中加载 PyTorch 的 NPZ？

```python
import numpy as np
import paddle

# 加载 NumPy 数组
data = np.load('pytorch_results.npz', allow_pickle=True)
encoder_output = data['vae_encoder'].item()['encoder_output']

# 转换为 Paddle Tensor
paddle_tensor = paddle.to_tensor(encoder_output)
```

### Q3: NPZ 文件太大怎么办？

```python
# 使用压缩保存
np.savez_compressed('output.npz', data=large_array)

# 或使用 float16 减小精度
data_float16 = data.astype(np.float16)
np.savez('output.npz', data=data_float16)
```

### Q4: 如何合并多个 NPZ 文件？

```python
import numpy as np

# 加载多个文件
data1 = np.load('file1.npz', allow_pickle=True)
data2 = np.load('file2.npz', allow_pickle=True)

# 合并
merged = {}
for key in data1.keys():
    merged[key] = data1[key]
for key in data2.keys():
    merged[key] = data2[key]

# 保存
np.savez('merged.npz', **merged)
```

## 总结

| 需求 | 命令 |
|------|------|
| 快速查看 | `python view_npz.py file.npz` |
| 查看数值 | `python view_npz.py file.npz --detail` |
| 对比精度 | `python view_npz.py file1.npz --compare file2.npz` |
| 导出文本 | `python view_npz.py file.npz --export output.txt` |
| Python读取 | `np.load('file.npz', allow_pickle=True)` |

**推荐工作流**：
1. 用 `view_npz.py` 快速查看文件内容
2. 用 `--detail` 检查具体数值
3. 用 `--compare` 对比不同框架的精度
4. 用 `--export` 导出给非Python用户
