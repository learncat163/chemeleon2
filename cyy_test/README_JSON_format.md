# JSON格式精度测试使用指南

## 概述

更新后的精度测试工具使用**JSON格式**进行输入输出，方便AI工具读取和分析。

## 主要改进

### 1. JSON格式输出 ✨
- **主文件**: `pytorch_results.json` - 完整的JSON格式结果
- **易读版本**: `pytorch_results_readable.txt` - 人类可读的文本版本
- **日志文件**: `precision_test_log.txt` - 详细的执行日志

### 2. 详细的输入描述 📝
每个模块都包含完整的输入信息：
- 输入来源（固定结构/随机生成）
- 输入张量的详细信息（shape, dtype, 统计值）
- 对于小张量（≤20个元素），保存完整值
- 对于大张量，保存前4个值的详细信息

### 3. 多维数据处理 🔢
- 所有多维张量只输出**前4个维度的具体值**
- JSON格式存储为列表：`[value1, value2, value3, value4]`
- 保留完整的统计信息：mean, std, min, max

### 4. 兼容不同数据类型 ⚙️
- 浮点数：计算 mean, std, min, max
- 整数：只计算 min, max
- 自动处理各种torch.dtype

## JSON文件结构

```json
{
  "test_info": {
    "framework": "PyTorch",
    "test_name": "Chemeleon2 精度测试",
    "test_time": "2026-02-02 19:18:37",
    "random_seed": 42,
    "device": "cuda",
    "description": "用于框架迁移的精度对比测试"
  },
  "model_info": {
    "vae_checkpoint": "...",
    "ldm_checkpoint": "...",
    "structure_file": "..."
  },
  "modules": {
    "vae_encoder": {
      "module_name": "vae_encoder",
      "description": "VAE编码器：输入晶体结构，输出潜在向量",
      "input": {
        "input_source": "fixed_structure",
        "structure_info": "Dy1 Ho1 Er3 Tm1 Te3 As1 P2",
        "batch_info": {
          "num_graphs": 1,
          "total_atoms": 12
        },
        "input_tensors": {
          "atom_types": {
            "shape": [12],
            "dtype": "torch.int64",
            "first_4_values": [66.0, 67.0, 68.0, 68.0],
            "all_values": [66, 67, 68, 68, 68, 69, 52, 52, 52, 33, 15, 15],
            "min": 15,
            "max": 69
          },
          "frac_coords": {
            "shape": [12, 3],
            "dtype": "torch.float32",
            "first_4_values": [0.1694, 0.2519, 0.4532, 0.4915],
            "mean": 0.49882931,
            "std": 0.28824043,
            "min": 0.00315657,
            "max": 0.99026763
          }
        }
      },
      "output": {
        "encoder_output": {
          "shape": [12, 512],
          "dtype": "torch.float32",
          "first_4_values": [-0.1799, -0.0465, -0.2934, -0.0578],
          "mean": 0.01894787,
          "std": 0.94812447,
          "min": -5.15007877,
          "max": 15.99389553
        },
        "latent_mean": { ... },
        "latent_logvar": { ... }
      }
    },
    "vae_decoder": { ... },
    "vae_full": { ... },
    "ldm_denoiser": { ... },
    "ldm_sampling": { ... }
  }
}
```

## 使用方法

### 1. 运行测试

```bash
cd /home/cao/opensource/cailiao/chemeleon2

# 运行PyTorch测试
.venv/bin/python cyy_test/test_precision.py

# 输出文件：
# - cyy_test/outputs/precision_test/pytorch_results.json (JSON格式)
# - cyy_test/outputs/precision_test/pytorch_results_readable.txt (易读文本)
# - cyy_test/outputs/precision_test/precision_test_log.txt (执行日志)
```

### 2. 对比结果

#### 方法1: 使用对比工具（推荐）

```bash
# 对比所有模块
.venv/bin/python cyy_test/compare_precision.py \
    cyy_test/outputs/precision_test/pytorch_results.json \
    paddle_results.json

# 只对比特定模块
.venv/bin/python cyy_test/compare_precision.py \
    cyy_test/outputs/precision_test/pytorch_results.json \
    paddle_results.json \
    --module vae_encoder
```

#### 方法2: 使用AI工具分析

JSON格式非常适合被AI工具读取和分析：

```python
import json

# 读取结果
with open('pytorch_results.json', 'r') as f:
    pytorch_data = json.load(f)

with open('paddle_results.json', 'r') as f:
    paddle_data = json.load(f)

# 提取特定模块的输出
vae_encoder_pytorch = pytorch_data['modules']['vae_encoder']['output']
vae_encoder_paddle = paddle_data['modules']['vae_encoder']['output']

# 对比latent_mean的前4个值
pytorch_values = vae_encoder_pytorch['latent_mean']['first_4_values']
paddle_values = vae_encoder_paddle['latent_mean']['first_4_values']

# 计算差异
diffs = [abs(p - pd) for p, pd in zip(pytorch_values, paddle_values)]
print(f"最大差异: {max(diffs):.8f}")
```

#### 方法3: 文本对比

```bash
# 对比易读文本版本
diff cyy_test/outputs/precision_test/pytorch_results_readable.txt \
     paddle_results_readable.txt
```

## 模块说明

### vae_encoder
**输入**: 晶体结构（atom_types, frac_coords, lattices）
**输出**: 
- encoder_output: 编码器输出 [N, 512]
- quant_conv_output: 量化卷积输出 [N, 16]
- latent_mean: 潜在向量均值 [N, 8]
- latent_logvar: 潜在向量对数方差 [N, 8]
- latent_z_sampled: 采样后的潜在向量 [N, 8]

### vae_decoder
**输入**: 潜在向量 z [N, 8]
**输出**: 重建的晶体结构
- reconstructed_atom_types: [N, 100]
- reconstructed_frac_coords: [N, 3]
- reconstructed_lengths: [1, 3]
- reconstructed_angles: [1, 3]

### vae_full
**输入**: 晶体结构
**输出**: 
- latent_mean, latent_logvar
- reconstruction_metrics: coord_mae, lattice_mae

### ldm_denoiser
**输入**: 
- input_z: 噪声潜在向量 [2, 10, 8]
- input_t: 时间步 [13, 25]
**输出**: 
- predicted_noise: 预测的噪声 [2, 20, 8]

### ldm_sampling
**输入**: 参考结构（用于确定原子数）
**输出**: 
- generated_frac_coords: 生成的坐标
- generated_lattices: 生成的晶格
- generated_structures: 生成结构的组成

## 输入数据固定性

所有测试使用固定输入保证可复现：

1. **随机种子**: 42
2. **VAE输入**: 固定测试结构 (Dy1 Ho1 Er3 Tm1 Te3 As1 P2)
3. **LDM输入**: 
   - z: `torch.randn(2, 10, 8)` with seed=42
   - t: 固定时间步 [13, 25]

## 对比工具功能

对比工具支持：
- ✅ 自动识别JSON和TXT格式
- ✅ 递归对比嵌套的字典结构
- ✅ 处理列表和单值
- ✅ 精度等级评估（完美/优秀/良好/一般/差）
- ✅ 计算MAE、Max Diff、相对误差
- ✅ 支持模块筛选

## 精度等级标准

- **✅ 完美**: Diff < 1e-6
- **✅ 优秀**: Diff < 1e-5
- **✅ 良好**: Diff < 1e-4
- **⚠️  一般**: Diff < 1e-3
- **❌ 差**: Diff > 1e-3

## AI工具读取示例

### 使用Python读取

```python
import json

def load_test_results(json_path):
    """加载测试结果"""
    with open(json_path, 'r') as f:
        return json.load(f)

def extract_module_output(data, module_name):
    """提取特定模块的输出"""
    return data['modules'][module_name]['output']

def compare_first_4_values(pytorch_data, paddle_data, module_name, tensor_name):
    """对比前4个值"""
    pytorch_values = pytorch_data['modules'][module_name]['output'][tensor_name]['first_4_values']
    paddle_values = paddle_data['modules'][module_name]['output'][tensor_name]['first_4_values']
    
    diffs = [abs(p - pd) for p, pd in zip(pytorch_values, paddle_values)]
    
    return {
        'max_diff': max(diffs),
        'mean_diff': sum(diffs) / len(diffs),
        'pytorch_values': pytorch_values,
        'paddle_values': paddle_values
    }

# 示例
pytorch_results = load_test_results('pytorch_results.json')
paddle_results = load_test_results('paddle_results.json')

comparison = compare_first_4_values(
    pytorch_results, 
    paddle_results, 
    'vae_encoder', 
    'latent_mean'
)

print(f"最大差异: {comparison['max_diff']:.8e}")
print(f"平均差异: {comparison['mean_diff']:.8e}")
```

### 使用jq命令行工具

```bash
# 提取VAE encoder的latent_mean
jq '.modules.vae_encoder.output.latent_mean' pytorch_results.json

# 提取所有模块的first_4_values
jq '.modules | to_entries | .[] | {module: .key, first_4: .value.output | to_entries | .[] | select(.key | endswith("first_4_values")) | .value}' pytorch_results.json

# 对比两个文件的测试信息
jq '.test_info' pytorch_results.json
jq '.test_info' paddle_results.json
```

## 注意事项

1. **JSON文件大小**: 约16KB，包含所有详细信息
2. **精度**: 所有浮点数保存8位小数
3. **数组长度**: 大张量只保存前4个值，完整统计信息保留
4. **兼容性**: 对比工具向后兼容旧的TXT格式
5. **编码**: 文件使用UTF-8编码，中文完全支持

## 文件对比

| 特性 | JSON格式 | 旧TXT格式 |
|------|---------|----------|
| AI工具读取 | ✅ 优秀 | ⚠️  需要解析 |
| 人类阅读 | ⚠️  需要工具 | ✅ 直接 |
| 结构化数据 | ✅ 完美 | ❌ 扁平 |
| 输入信息 | ✅ 详细 | ❌ 无 |
| 嵌套数据 | ✅ 支持 | ❌ 不支持 |
| 文件大小 | 16KB | 6KB |

## 最佳实践

1. **开发调试**: 使用 `pytorch_results_readable.txt` 快速查看
2. **自动化对比**: 使用 `pytorch_results.json` 进行程序化分析
3. **详细日志**: 查看 `precision_test_log.txt` 了解执行细节
4. **模块对比**: 使用 `--module` 参数逐个模块排查问题
5. **AI分析**: 直接将JSON文件提供给AI工具进行深度分析
