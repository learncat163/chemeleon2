import torch
import numpy as np
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.ldm_module.denoisers.dit import DiT, get_pos_embedding

checkpoint_path = project_root / 'checkpoints/v0.0.1/alex_mp_20/ldm/ldm_rl_dng_tuor5vgd.ckpt'
input_data_path = Path(__file__).parent / 'outputs/precision_test/pytorch_results.npz'
output_path = Path(__file__).parent / 'outputs/precision_test/denoiser_layers.npz'

print("Loading checkpoint...")
checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
state_dict = checkpoint['state_dict']

print("Creating model...")
denoiser = DiT(
    latent_dim=8,
    hidden_size=768,
    num_heads=8,
    depth=12,
    mlp_ratio=4.0,
    learn_sigma=True
)

ldm_state = {}
for key, value in state_dict.items():
    if key.startswith('ldm.denoiser.'):
        new_key = key.replace('ldm.denoiser.', '')
        ldm_state[new_key] = value

print(f"Loading {len(ldm_state)} parameters...")
denoiser.load_state_dict(ldm_state, strict=False)
denoiser.eval()

print("Loading input data...")
input_data = np.load(input_data_path, allow_pickle=True)
ldm_data = input_data['ldm_denoiser'].item()
denoiser_input_z = torch.from_numpy(ldm_data['denoiser_input_z']).float()
denoiser_input_t = torch.from_numpy(ldm_data['denoiser_input_t']).long()

batch_size, num_atoms, latent_dim = denoiser_input_z.shape
print(f"Input shape: {denoiser_input_z.shape}")
print(f"Timesteps: {denoiser_input_t}")

mask = torch.ones(batch_size, num_atoms, dtype=torch.bool)

layers_output = {}

print("\nForward pass with layer-by-layer saving...")

with torch.no_grad():
    token_indices = torch.cumsum(mask.long(), dim=-1) - 1
    pos_emb = get_pos_embedding(token_indices, denoiser.hidden_size)
    
    x = denoiser.x_embedder(denoiser_input_z) + pos_emb
    layers_output['after_x_embedder_pos'] = x.cpu().numpy()
    print(f"after_x_embedder_pos: shape={x.shape}, mean={x.mean():.6f}, std={x.std():.6f}")
    
    c = denoiser.t_embedder(denoiser_input_t)
    layers_output['after_t_embedder'] = c.cpu().numpy()
    print(f"after_t_embedder: shape={c.shape}, mean={c.mean():.6f}, std={c.std():.6f}")
    
    if hasattr(denoiser, 'y_embedder') and denoiser.y_embedder is not None:
        y_emb = torch.zeros(x.shape[0], denoiser.hidden_size, dtype=x.dtype)
        c = c + y_emb
        layers_output['after_y_embedder'] = c.cpu().numpy()
    
    mask_inverted = ~mask
    
    for i, block in enumerate(denoiser.blocks):
        x = block(x, c, mask_inverted)
        layers_output[f'after_block_{i}'] = x.cpu().numpy()
        if i < 3 or i >= len(denoiser.blocks) - 2:
            print(f"after_block_{i}: shape={x.shape}, mean={x.mean():.6f}, std={x.std():.6f}")
    
    x = denoiser.final_layer(x, c)
    layers_output['after_final_layer'] = x.cpu().numpy()
    print(f"after_final_layer: shape={x.shape}, mean={x.mean():.6f}, std={x.std():.6f}")
    
    assert x.shape[2] == 2 * denoiser.latent_dim
    x = x.reshape(x.shape[0], 2 * x.shape[1], denoiser.latent_dim)
    layers_output['after_reshape'] = x.cpu().numpy()
    print(f"after_reshape: shape={x.shape}, mean={x.mean():.6f}, std={x.std():.6f}")
    
    x = x * mask.tile(2).unsqueeze(-1)
    layers_output['final_output'] = x.cpu().numpy()
    print(f"final_output: shape={x.shape}, mean={x.mean():.6f}, std={x.std():.6f}")

print(f"\nSaving {len(layers_output)} layer outputs to {output_path}")
np.savez(output_path, **layers_output)
print("Done!")
