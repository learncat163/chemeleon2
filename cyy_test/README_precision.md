# 精度测试说明 - 用于框架迁移对比

## 概述

`test_precision.py` 是一个专门用于模型精度测试和对比的工具，主要用途是：

1. **输出不同模型组件的中间结果**（VAE Encoder/Decoder, LDM Denoiser）
2. **保存数值用于精度对比**（便于与 Paddle 等其他框架对比）
3. **固定随机种子**确保结果可复现

## 测试内容

脚本测试以下5个模块：

### 1. VAE Encoder
- **输入**: 晶体结构 (12个原子)
- **输出**:
  - `encoder_output`: (12, 512) - Transformer Encoder 输出
  - `quant_conv_output`: (12, 16) - 量化卷积后
  - `latent_mean`: (12, 8) - 潜在空间均值
  - `latent_logvar`: (12, 8) - 潜在空间对数方差
  - `latent_z`: (12, 8) - 采样的潜在向量

### 2. VAE Decoder
- **输入**: 潜在向量 z (12, 8)
- **输出**:
  - `decoder_post_quant`: (12, 512) - 反量化卷积后
  - `decoder_output_atom_types`: (12, 100) - 原子类型logits
  - `decoder_output_frac_coords`: (12, 3) - 分数坐标
  - `decoder_output_lengths`: (1, 3) - 晶格长度
  - `decoder_output_angles`: (1, 3) - 晶格角度

### 3. VAE 完整流程 (Encode + Decode)
- **测试**: 编码-解码重建误差
- **输出**:
  - `original_coords`: (12, 3) - 原始坐标
  - `reconstructed_coords`: (12, 3) - 重建坐标
  - 坐标 MAE: ~0.002
  - 晶格 MAE: ~0.008

### 4. LDM Denoiser
- **输入**: 
  - `z`: (2, 10, 8) - 潜在向量
  - `t`: (2,) - 时间步 [13, 25]
- **输出**:
  - `denoiser_output`: (2, 20, 8) - 预测的噪声

### 5. LDM 完整采样
- **测试**: 从噪声生成新结构
- **输出**:
  - `generated_coords`: (12, 3) - 生成的坐标
  - `generated_lattices`: (1, 3, 3) - 生成的晶格

## 使用方法

### 运行测试

```bash
cd /home/cao/opensource/cailiao/chemeleon2
.venv/bin/python cyy_test/test_precision.py
```

**输出文件**：
- `cyy_test/outputs/precision_test/pytorch_results.npz` - 数值结果（62KB）
- `cyy_test/outputs/precision_test/precision_test_log.txt` - 详细日志（5.7KB，214行）

所有控制台输出都会同时保存到日志文件中，方便日后查看和对比。

### 查看结果

测试结果保存在 `cyy_test/outputs/precision_test/pytorch_results.npz`:

```python
import numpy as np

# 加载结果
results = np.load('cyy_test/outputs/precision_test/pytorch_results.npz', allow_pickle=True)

# 查看所有模块
print('可用模块:', list(results.keys()))
# ['vae_encoder', 'vae_decoder', 'vae_full', 'ldm_denoiser', 'ldm_sampling']

# 查看 VAE Encoder 的输出
vae_encoder_results = results['vae_encoder'].item()
print('Encoder 输出键:', vae_encoder_results.keys())
print('Latent mean shape:', vae_encoder_results['latent_mean'].shape)  # (12, 8)
print('Latent mean 前4个值:', vae_encoder_results['latent_mean'].flatten()[:4])

# 查看 LDM Denoiser 的输出
ldm_results = results['ldm_denoiser'].item()
print('Denoiser 输入 shape:', ldm_results['denoiser_input_z'].shape)  # (2, 10, 8)
print('Denoiser 输出 shape:', ldm_results['denoiser_output'].shape)   # (2, 20, 8)
print('Denoiser 输出前4个值:', ldm_results['denoiser_output'].flatten()[:4])
```

**查看日志文件**：

```bash
# 使用便捷脚本（推荐）
./cyy_test/view_precision_log.sh

# 或直接查看文件
cat cyy_test/outputs/precision_test/precision_test_log.txt

# 查看前50行
head -50 cyy_test/outputs/precision_test/precision_test_log.txt

# 搜索特定模块
grep -A 20 "测试 VAE Encoder" cyy_test/outputs/precision_test/precision_test_log.txt
```

**日志查看器选项**：
1. 查看完整日志（214行）
2. 查看 VAE Encoder 部分
3. 查看 VAE Decoder 部分  
4. 查看 LDM Denoiser 部分
5. 查看测试摘要
6. 查看所有张量统计信息
7. 查看前50行
8. 查看最后50行

示例：`./cyy_test/view_precision_log.sh 5` 查看测试摘要

## 输出格式说明

每个张量都会输出以下信息：

```
Encoder Output (x):
  Shape: (12, 512)           # 张量形状
  Dtype: torch.float32       # 数据类型
  Device: cuda:0            # 设备
  Mean: 0.018948            # 均值
  Std: 0.948124             # 标准差
  Min: -5.150079            # 最小值
  Max: 15.993896            # 最大值
  First 4 values: [-0.179900 -0.046454 -0.293423 -0.057796]  # 前4个值
```

**注意**: 如果维度超过4，只显示前4个值，避免输出过长。

## 精度对比流程

### 1. PyTorch 侧（已完成）

运行 `test_precision.py` 生成 `pytorch_results.npz`

### 2. Paddle 侧（待实现）

创建类似的测试脚本：

```python
# paddle_test_precision.py
import paddle
import numpy as np

# 1. 加载转换后的 Paddle 模型
vae = load_paddle_vae('checkpoints/paddle/vae.pdparams')
ldm = load_paddle_ldm('checkpoints/paddle/ldm.pdparams')

# 2. 使用相同的随机种子
paddle.seed(42)
np.random.seed(42)

# 3. 加载 PyTorch 的输入数据
pytorch_results = np.load('pytorch_results.npz', allow_pickle=True)

# 4. 在 Paddle 中使用相同的输入
vae_encoder_input = ...  # 从 PyTorch 结果中获取
paddle_encoder_output = vae.encoder(vae_encoder_input)

# 5. 保存 Paddle 结果
np.savez('paddle_results.npz', 
         vae_encoder={'encoder_output': paddle_encoder_output.numpy()},
         ...)
```

### 3. 精度对比

```python
import numpy as np

# 加载两个框架的结果
pytorch_results = np.load('pytorch_results.npz', allow_pickle=True)
paddle_results = np.load('paddle_results.npz', allow_pickle=True)

# 对比 VAE Encoder
pt_encoder = pytorch_results['vae_encoder'].item()['encoder_output']
pd_encoder = paddle_results['vae_encoder'].item()['encoder_output']

# 计算差异
mae = np.abs(pt_encoder - pd_encoder).mean()
mse = np.square(pt_encoder - pd_encoder).mean()
max_diff = np.abs(pt_encoder - pd_encoder).max()
relative_error = mae / np.abs(pt_encoder).mean()

print(f'VAE Encoder 精度对比:')
print(f'  MAE: {mae:.6e}')
print(f'  MSE: {mse:.6e}')
print(f'  Max Diff: {max_diff:.6e}')
print(f'  Relative Error: {relative_error:.6%}')

# 对比所有模块
for module in ['vae_encoder', 'vae_decoder', 'vae_full', 'ldm_denoiser', 'ldm_sampling']:
    print(f'\n{module}:')
    pt_data = pytorch_results[module].item()
    pd_data = paddle_results[module].item()
    
    for key in pt_data.keys():
        if key in pd_data:
            pt_tensor = pt_data[key]
            pd_tensor = pd_data[key]
            mae = np.abs(pt_tensor - pd_tensor).mean()
            print(f'  {key}: MAE = {mae:.6e}')
```

## 可接受的精度差异

不同框架之间的数值精度差异通常来自：

1. **浮点运算顺序**：不同框架的运算顺序可能不同
2. **数学函数实现**：如 softmax, layer_norm 的实现细节
3. **随机数生成**：即使相同种子，不同框架的随机数也可能不同

### 建议的阈值

- **MAE < 1e-5**: 优秀（基本等价）
- **MAE < 1e-4**: 良好（可接受）
- **MAE < 1e-3**: 一般（需要检查）
- **MAE > 1e-3**: 差（可能有bug）

### 相对误差

- **Relative Error < 0.1%**: 优秀
- **Relative Error < 1%**: 良好
- **Relative Error < 5%**: 可接受
- **Relative Error > 5%**: 需要调查

## 重要特性

### 1. 固定随机种子

```python
def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
```

确保每次运行结果完全一致，便于对比。

### 2. 只显示前4维

```python
def print_tensor_info(tensor, name, max_dims=4):
    # ...
    flat = tensor.flatten()
    show_n = min(max_dims, len(flat))
    print(f"  First {show_n} values: {flat[:show_n].cpu().numpy()}")
```

避免高维张量输出过长。

### 3. 详细的统计信息

每个张量输出：形状、类型、设备、均值、标准差、最小值、最大值、前N个值。

## 常见问题

### Q1: 为什么 Denoiser 输出是 (2, 20, 8) 而输入是 (2, 10, 8)?

A: DiT (Diffusion Transformer) 使用了特殊的 token 扩展机制，输出维度可能不同于输入。

### Q2: decoder_output_atom_types 为什么是 (12, 100)?

A: 这是 logits (未归一化的概率)，100 是最大元素数量。需要通过 softmax + argmax 得到实际的原子类型。

### Q3: 为什么 Latent Z 的形状是 (12, 8)?

A: 12 是原子数量，8 是潜在空间维度 (latent_dim=8)。

### Q4: 如何确保 Paddle 使用相同的输入？

A: 
1. 保存 PyTorch 的原始输入到文件
2. 在 Paddle 中加载相同的输入
3. 确保数据类型和设备一致

## 后续步骤

1. ✅ **完成 PyTorch 测试** - 生成基准结果
2. ⏳ **转换模型权重** - PyTorch → Paddle
3. ⏳ **实现 Paddle 测试** - 使用相同输入
4. ⏳ **精度对比分析** - 逐层对比误差
5. ⏳ **调优和验证** - 确保精度达标

## 参考资料

- [PyTorch 模型结构](../cyy_test/print_model_structure.py)
- [基础功能测试](../cyy_test/three.py)
- [VAE 模块源码](../src/vae_module/)
- [LDM 模块源码](../src/ldm_module/)
