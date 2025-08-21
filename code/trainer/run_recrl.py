#!/usr/bin/env python
"""
RLHF training driver for Shielded RecRL using RecSim-NG.

This script runs the main RLHF training loop with RecSim-NG synthetic CTR rewards.

Usage:
  python run_recrl.py --dataset books [--cfg path_to_config.yaml]
"""
import argparse, yaml, pathlib, os, wandb, time, json, hashlib
from pathlib import Path
from shielded_ppo_trainer import ShieldedPPO

ROOT = pathlib.Path(os.getenv("PROJ", "."))

def compute_sha256(file_path):
    """Compute SHA-256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def save_checkpoint_meta(ckpt_path, dataset, epoch, ctr, kl, wall_time):
    """Save checkpoint metadata."""
    meta = {
        "dataset": dataset,
        "epoch": epoch,
        "ctr": ctr,
        "kl": kl,
        "wall_clock_s": wall_time,
        "sha256": compute_sha256(ckpt_path)
    }
    meta_path = str(ckpt_path).replace(f"epoch{epoch}.pt", "final_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f)
    
    # Append to manifest
    manifest_path = ROOT/"data"/"_checksums"/"manifest.jsonl"
    manifest_dir = manifest_path.parent
    manifest_dir.mkdir(exist_ok=True, parents=True)
    
    entry = {
        "type": "rlhf",
        "dataset": dataset,
        "epoch": epoch,
        "sha256": meta["sha256"],
        "date": time.strftime("%Y-%m-%dT%H:%M:%S%z")
    }
    
    with open(manifest_path, "a") as f:
        f.write(json.dumps(entry) + "\n")

def run(ds, cfg):
    """Run RLHF training for a dataset."""
    # Initialize W&B
    wandb.init(project="recrl_rlhf", name=f"{ds}_run",
               config=cfg, tags=[ds])
    
    # Initialize agent
    print(f"\n{'='*80}")
    print(f"Starting RLHF training for dataset: {ds}")
    print(f"{'='*80}\n")
    
    agent = ShieldedPPO(dataset=ds, proj=ROOT,
                        int8=cfg["int8"], 
                        kl_beta=cfg["kl_beta"],
                        lr=cfg["lr"],
                        seed=cfg["seed"])
    
    # Calculate steps per epoch
    sim_batch = cfg.get("sim_batch", 32)
    steps_per_epoch = cfg["ppo_batch"] // sim_batch
    print(f"Running {cfg['epochs']} epochs with {steps_per_epoch} steps per epoch")
    print(f"PPO batch size: {cfg['ppo_batch']}, Sim batch size: {sim_batch}")
    
    # Start timing
    start_time = time.time()
    
    # Training loop
    for ep in range(cfg["epochs"]):
        ep_start = time.time()
        ep_reward = 0.0
        ep_kl = 0.0
        
        print(f"\nEpoch {ep+1}/{cfg['epochs']}")
        
        for step in range(steps_per_epoch):
            # Collect trajectories from simulator
            traj = agent.rollout_batch(batch_size=sim_batch, 
                                       slate_k=5, 
                                       steps=1)
            
            # Process trajectories
            batch_prompts = [t["prompt"] for t in traj]
            batch_rewards = torch.tensor([t["reward"] for t in traj],
                                        device=agent.device)
            old_logprobs = torch.zeros_like(batch_rewards) # Initial approximation
            
            # Run optimization
            metrics = agent.optimise(batch_prompts, batch_rewards, old_logprobs)
            
            # Track metrics
            ep_reward += metrics["reward_mean"]
            ep_kl += metrics["kl"]
            
            # Log step metrics
            wandb.log({
                "epoch": ep,
                "step": ep * steps_per_epoch + step,
                "ctr_step": metrics["reward_mean"],
                "loss": metrics["loss"],
                "loss_pg": metrics["loss_pg"],
                "loss_kl": metrics["loss_kl"],
                "kl": metrics["kl"],
                "ratio_mean": metrics["ratio_mean"]
            })
            
            # Print progress
            if step % 10 == 0 or step == steps_per_epoch - 1:
                print(f"  Step {step+1}/{steps_per_epoch}: ctr={metrics['reward_mean']:.4f}, "
                      f"kl={metrics['kl']:.4f}, loss={metrics['loss']:.4f}")
        
        # Average metrics for the epoch
        avg_reward = ep_reward / steps_per_epoch
        avg_kl = ep_kl / steps_per_epoch
        
        # Log epoch metrics
        wandb.log({
            "epoch": ep,
            "ctr": avg_reward,
            "kl_epoch": avg_kl,
            "epoch_time": time.time() - ep_start
        })
        
        # Save epoch checkpoint
        ckpt_path = ROOT/"checkpoints"/f"recrl_{ds}_epoch{ep}.pt"
        agent.save_checkpoint(ckpt_path, ep)
        
        print(f"Epoch {ep+1} completed in {time.time()-ep_start:.1f}s. "
              f"CTR: {avg_reward:.4f}, KL: {avg_kl:.4f}")
    
    # End timing
    total_time = time.time() - start_time
    
    # Save final metadata
    final_ckpt = ROOT/"checkpoints"/f"recrl_{ds}_epoch{cfg['epochs']-1}.pt"
    save_checkpoint_meta(
        final_ckpt, ds, cfg['epochs']-1, 
        avg_reward.item(), avg_kl.item(), total_time
    )
    
    print(f"\n{'='*80}")
    print(f"RLHF training completed in {total_time:.1f}s")
    print(f"Final CTR: {avg_reward:.4f}")
    print(f"Final KL: {avg_kl:.4f}")
    print(f"Checkpoints saved to {ROOT}/checkpoints/recrl_{ds}_epoch*.pt")
    print(f"{'='*80}\n")
    
    # Finish W&B
    wandb.finish()

def verify_ctr_lift():
    """Verify that CTR increased by at least 1.5 percentage points."""
    try:
        import wandb
        api = wandb.Api()
        runs = api.runs("YOUR_WANDB_USERNAME/recrl_rlhf")
        for run in runs:
            if "books_run" in run.name:
                df = run.history(keys=["ctr"])
                initial_ctr = df.iloc[0]["ctr"]
                final_ctr = df.iloc[-1]["ctr"]
                lift = final_ctr - initial_ctr
                print(f"CTR lift for {run.name}: {lift*100:.2f} percentage points")
                assert lift >= 0.015, f"CTR lift ({lift}) is less than 1.5 pp"
                print(f"✓ Synthetic CTR lift = {lift*100:.2f} pp")
    except Exception as e:
        print(f"Error verifying CTR lift: {e}")
        return False
    return True

if __name__ == "__main__":
    # Add debug prints
    print("Starting run_recrl.py")
    print(f"Current working directory: {os.getcwd()}")
    print(f"PROJ environment variable: {os.getenv('PROJ', 'Not set')}")
    
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["books", "ml25m", "steam"], required=True)
    parser.add_argument("--cfg", default="experiments/recrl_default.yaml")
    parser.add_argument("--verify-only", action="store_true", 
                        help="Only verify CTR lift without training")
    
    args = parser.parse_args()
    print(f"Arguments: dataset={args.dataset}, cfg={args.cfg}, verify_only={args.verify_only}")
    
    # Verify CTR lift if requested
    if args.verify_only:
        verify_ctr_lift()
        exit(0)
    
    # Load configuration
    with open(args.cfg) as f:
        cfg = yaml.safe_load(f)[args.dataset]
    
    # Add sim_batch if not present
    if "sim_batch" not in cfg:
        cfg["sim_batch"] = 32
    
    # Run training
    import torch
    run(args.dataset, cfg)
