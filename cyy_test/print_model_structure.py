"""
打印 Chemeleon2 模型结构

简单打印 VAE 和 LDM 模型的结构
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.vae_module.vae_module import VAEModule
from src.ldm_module.ldm_module import LDMModule


def main():
    """主函数"""
    # 默认模型路径
    alex_vae_path = "checkpoints/v0.0.1/alex_mp_20/vae/dng_j1jgz9t0_v1.ckpt"
    alex_ldm_path = "checkpoints/v0.0.1/alex_mp_20/ldm/ldm_rl_dng_tuor5vgd.ckpt"
    
    # 打印 VAE 模型
    print("=" * 80)
    print("VAE 模型结构")
    print("=" * 80)
    vae = VAEModule.load_from_checkpoint(alex_vae_path, weights_only=False)
    print(vae)
    
    print("\n" * 2)
    
    # 打印 LDM 模型
    print("=" * 80)
    print("LDM 模型结构")
    print("=" * 80)
    ldm = LDMModule.load_from_checkpoint(alex_ldm_path, vae_ckpt_path=alex_vae_path, weights_only=False)
    print(ldm)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
