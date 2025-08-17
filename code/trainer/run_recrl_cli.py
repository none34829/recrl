#!/usr/bin/env python
"""
RLHF training driver for Shielded RecRL using RecSim-NG - Command Line Interface.

This script runs the main RLHF training loop with RecSim-NG synthetic CTR rewards.
Accepts command-line arguments for easy automation.

Usage:
  python run_recrl_cli.py --dataset books --ranker_ckpt path/to/ranker.pt --projection_basis path/to/basis.pt
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

def run_cli(args):
    """Run RLHF training with command-line arguments."""
    # Initialize W&B
    wandb.init(project="recrl_rlhf", name=f"{args.dataset}_{args.tag}",
               config=vars(args), tags=[args.dataset, args.tag])
    
    # Initialize agent
    print(f"\n{'='*80}")
    print(f"Starting RLHF training for dataset: {args.dataset}")
    print(f"Tag: {args.tag}")
    print(f"Ranker checkpoint: {args.ranker_ckpt}")
    print(f"Projection basis: {args.projection_basis}")
    print(f"{'='*80}\n")
    
    agent = ShieldedPPO(
        dataset=args.dataset, 
        proj=ROOT,
        int8=args.int8, 
        kl_beta=args.kl_beta,
        lr=args.lr,
        seed=args.seed,
        ranker_ckpt=args.ranker_ckpt,
        projection_basis=args.projection_basis,
        no_projection=args.no_projection
    )
    
    # Calculate steps per epoch
    sim_batch = args.sim_batch
    steps_per_epoch = args.ppo_batch // sim_batch
    print(f"Running {args.epochs} epochs with {steps_per_epoch} steps per epoch")
    print(f"PPO batch size: {args.ppo_batch}, Sim batch size: {sim_batch}")
    
    # Start timing
    start_time = time.time()
    
    # Training loop
    for ep in range(args.epochs):
        ep_start = time.time()
        ep_reward = 0.0
        ep_kl = 0.0
        
        print(f"\nEpoch {ep+1}/{args.epochs}")
        
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
            
            # Accumulate metrics
            ep_reward += metrics["reward_mean"].item()
            ep_kl += metrics["kl"].item()
            
            # Log step metrics
            wandb.log({
                "step": ep * steps_per_epoch + step,
                "ctr_step": metrics["reward_mean"],
                "kl_step": metrics["kl"],
                "loss": metrics["loss"],
                "loss_pg": metrics["loss_pg"],
                "loss_kl": metrics["loss_kl"],
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
        ckpt_path = ROOT/"checkpoints"/f"recrl_{args.dataset}_{args.tag}_epoch{ep}.pt"
        agent.save_checkpoint(ckpt_path, ep)
        
        print(f"Epoch {ep+1} completed in {time.time()-ep_start:.1f}s. "
              f"CTR: {avg_reward:.4f}, KL: {avg_kl:.4f}")
    
    # End timing
    total_time = time.time() - start_time
    
    # Save final metadata
    final_ckpt = ROOT/"checkpoints"/f"recrl_{args.dataset}_{args.tag}_epoch{args.epochs-1}.pt"
    save_checkpoint_meta(
        final_ckpt, args.dataset, args.epochs-1, 
        avg_reward, avg_kl, total_time
    )
    
    print(f"\n{'='*80}")
    print(f"RLHF training completed in {total_time:.1f}s")
    print(f"Final CTR: {avg_reward:.4f}")
    print(f"Final KL: {avg_kl:.4f}")
    print(f"Checkpoints saved to {ROOT}/checkpoints/recrl_{args.dataset}_{args.tag}_epoch*.pt")
    print(f"{'='*80}\n")
    
    # Finish W&B
    wandb.finish()

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="Shielded RecRL CLI Training")
    
    # Required arguments
    parser.add_argument("--dataset", choices=["books", "ml25m", "steam"], required=True,
                        help="Dataset to train on")
    parser.add_argument("--ranker_ckpt", type=str, required=True,
                        help="Path to trained SASRec ranker checkpoint")
    parser.add_argument("--tag", type=str, default="default",
                        help="Tag for this run (used in checkpoint names)")
    
    # Optional arguments
    parser.add_argument("--projection_basis", type=str, default=None,
                        help="Path to projection basis matrix (optional if --no_projection)")
    parser.add_argument("--no_projection", action="store_true",
                        help="Disable gradient projection (ablation)")
    
    # Training parameters
    parser.add_argument("--epochs", type=int, default=8,
                        help="Number of training epochs")
    parser.add_argument("--ppo_batch", type=int, default=256,
                        help="PPO batch size")
    parser.add_argument("--sim_batch", type=int, default=32,
                        help="Simulation batch size")
    parser.add_argument("--kl_beta", type=float, default=0.05,
                        help="KL penalty coefficient")
    parser.add_argument("--lr", type=float, default=3e-5,
                        help="Learning rate")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--int8", action="store_true",
                        help="Use 8-bit quantization")
    
    # Generation parameters
    parser.add_argument("--max_new_tokens", type=int, default=40,
                        help="Maximum new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Generation temperature")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="Generation top_p")
    
    # LoRA parameters
    parser.add_argument("--lora_rank", type=int, default=16,
                        help="LoRA rank")
    parser.add_argument("--micro_batch_size", type=int, default=4,
                        help="Micro batch size for gradient accumulation")
    parser.add_argument("--grad_accum", type=int, default=2,
                        help="Gradient accumulation steps")
    parser.add_argument("--max_seq_len_explainer", type=int, default=384,
                        help="Maximum sequence length for explainer")
    parser.add_argument("--explanation_max_len", type=int, default=160,
                        help="Maximum explanation length")
    parser.add_argument("--max_steps", type=int, default=20000,
                        help="Maximum training steps")
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.no_projection and args.projection_basis is None:
        raise ValueError("--projection_basis is required unless --no_projection is set")
    
    # Run training
    import torch
    run_cli(args)
