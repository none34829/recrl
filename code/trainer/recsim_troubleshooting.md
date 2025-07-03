# RecSim-NG Integration Troubleshooting Guide

## Common Issues and Solutions

### TensorFlow CPU Warnings

**Symptom**: Warnings about AVX/SSE instructions when running RecSim-NG

**Cause**: CPU-only TensorFlow installation

**Solution**: These warnings are harmless and can be suppressed:
```bash
export TF_CPP_MIN_LOG_LEVEL=2
```

### Zero CTR Values

**Symptom**: CTR values remain constant at 0 during training

**Cause**: Explanation quality never passes the threshold required for positive rewards

**Solutions**:
- Reduce the token length threshold in the quality calculation
- Check that the explanations are being generated correctly
- Verify the prompts are properly formatted

### KL Divergence Explosion

**Symptom**: KL divergence increases dramatically after a few epochs

**Cause**: KL penalty coefficient (`kl_beta`) is too small

**Solutions**:
- Increase `kl_beta` to 0.1 or higher
- Implement ratio clipping to prevent extreme updates (e.g., clip to [0.1, 10])
- Add early stopping based on KL divergence

### Slow Simulation

**Symptom**: Simulation steps taking much longer than expected

**Cause**: TensorFlow operations running on GPU when not intended

**Solution**: Pin TensorFlow to CPU:
```bash
export CUDA_VISIBLE_DEVICES=0  # Only use first GPU for PyTorch
export TF_FORCE_GPU_ALLOW_GROWTH=true  # Limit TF memory use if it does use GPU
```

### RecSim-NG Import Errors

**Symptom**: ModuleNotFoundError for RecSim-NG modules

**Cause**: Missing or incomplete installation

**Solution**: Install with the correct extras:
```bash
pip install recsim-ng[tensorflow]==0.1.2
pip install tensorflow==2.15.0
```

### Slate Shape Mismatch

**Symptom**: Error about shape mismatch in slate or explanation quality tensors

**Cause**: Mismatch between batch_size and slate dimensions

**Solution**: Ensure consistent dimensions in:
1. Simulator initialization
2. State initialization
3. Slate generation
4. Explanation quality tensors

## Monitoring

### CTR Curve Analysis

A healthy training run should show:
- Gradual increase in CTR over epochs
- Minimal fluctuation between steps
- At least 1.5 percentage point increase from initial to final

### Memory Usage

Monitor GPU memory with:
```bash
watch -n 1 nvidia-smi
```

Expected usage:
- LLM (int8): ~16-18 GB
- TensorFlow: minimal (should be on CPU)

## Performance Tuning

### Batch Size Adjustment

If training is too slow:
- Increase `sim_batch` (e.g., to 64)
- Decrease steps_per_epoch accordingly

If hitting OOM errors:
- Decrease `sim_batch`
- Set `int8=True` in configuration

### Explanation Generation Speed

To speed up explanation generation:
- Lower `max_new_tokens` to 20-30
- Use higher `temperature` (0.8-0.9) for more diverse but potentially shorter outputs
- Implement batched generation if not already present

## Validation and Recovery

### Checkpoint Validation

Verify checkpoint integrity with:
```python
import torch
ck = torch.load('checkpoints/recrl_books_epoch0.pt', map_location='cpu')
print(list(ck.keys()))  # Should include 'model', 'optimizer', 'dataset', 'epoch'
```

### Run Recovery

If training crashes mid-run:
1. Identify the last valid checkpoint
2. Modify `run_recrl.py` to load the checkpoint and continue
3. Resume from the appropriate epoch
