# Shielded PPO Trainer Troubleshooting Guide

## Common Issues and Solutions

### Mixed Precision Errors

**Symptom**: `RuntimeError: expected scalar type Half but found Float`

**Cause**: Mixed dtypes between int-8 base model and LoRA FP32 adapters

**Solution**: Set `torch_dtype="auto"` in `load_base` function to ensure consistent precision

### Slow Gradient Projection

**Symptom**: Projection step takes a long time during training

**Cause**: Computing `Q @ (Q.t() @ grad)` on each step is inefficient

**Solution**: Pre-compute `QT = Q.t()` once during initialization and reuse it

### KL Divergence Exploding

**Symptom**: `loss_kl` value becomes very large during training

**Cause**: `kl_beta` coefficient is too small

**Solution**: Increase `kl_beta` to 0.05-0.1 to prevent the model from diverging too much

### CUDA Out of Memory

**Symptom**: `CUDA out of memory` error during training

**Cause**: Even with int-8 quantization, GPT-J-6B is still heavy on 12GB GPUs

**Solution**: 
- Set `max_new_tokens=20` to reduce generation length
- Use gradient checkpointing with `accelerate` library
- Reduce batch size
- Use `device_map="auto"` to distribute model across multiple GPUs

### Projection Not Working

**Symptom**: `check_projection.py` shows high projection components (>1e-5)

**Cause**: Incorrect implementation of projection or basis vectors

**Solution**:
- Verify that the basis file exists and has the correct shape
- Check that the projection formula is implemented correctly: `g_proj = grad - Q @ (QT @ grad)`
- Ensure the basis vectors are orthonormal

### Generation Hangs

**Symptom**: Model generation step hangs indefinitely

**Cause**: Using `bitsandbytes` + `device_map="auto"` on multi-GPU setups

**Solution**:
- Pin to a single GPU with `CUDA_VISIBLE_DEVICES=0`
- Or use `accelerate` to properly split the model

## Verification Steps

1. **Check Toy Run**: Verify that loss_pg decreases and KL remains stable in W&B logs

2. **Verify Projection**: Run `check_projection.py` to confirm max projection component < 1e-5

3. **Check LoRA Size**: The saved LoRA checkpoint should be around 26 MB, not GB

4. **Monitor VRAM**: Use `nvidia-smi` to ensure VRAM usage is ≤ 18 GB with int-8 quantization

## Advanced Debugging

### Gradient Flow Check

```python
import torch

# Inside the optimization loop, before projection
def check_grad_flow(model):
    named_params = {n: p for n, p in model.named_parameters() if p.requires_grad}
    avg_grad = {n: p.grad.abs().mean().item() if p.grad is not None else 0 
               for n, p in named_params.items()}
    print(f"Gradient flow: min={min(avg_grad.values()):.6f}, "
          f"max={max(avg_grad.values()):.6f}, "
          f"mean={sum(avg_grad.values())/len(avg_grad):.6f}")
    return avg_grad
```

### Projection Visualization

```python
import matplotlib.pyplot as plt
import numpy as np

def visualize_projection(Q, grad_flat):
    # Project grad onto Q basis
    proj = Q.t() @ grad_flat
    plt.figure(figsize=(10, 5))
    plt.bar(range(len(proj)), proj.abs().cpu().numpy())
    plt.title("Projection Components")
    plt.xlabel("Basis Vector Index")
    plt.ylabel("Absolute Projection Magnitude")
    plt.savefig("projection_viz.png")
```
