#!/usr/bin/env python
"""
Run a toy PPO loop to test the Shielded RecRL trainer.

This script runs a small PPO training loop to verify that:
1. Gradients flow correctly
2. The projection shield works
3. Loss decreases over time

Usage:
  python run_toy.py [--dataset books|ml25m|steam] [--int8] [--steps 30] [--no-wandb]
"""

import random, torch, os, json, pathlib, argparse, time
from pathlib import Path
from shielded_ppo_trainer import ShieldedPPO
from reward import click_reward

def main(args):
    # Initialize project path
    proj = Path(os.getenv("PROJ", "."))
    
    # Initialize W&B if enabled
    if not args.no_wandb:
        try:
            import wandb
            wandb.init(
                project="recrl_trainer",
                name=f"toy_{args.dataset}_seed{args.seed}",
                config={
                    "dataset": args.dataset,
                    "int8": args.int8,
                    "steps": args.steps,
                    "seed": args.seed,
                    "kl_beta": args.kl_beta,
                    "lr": args.lr
                }
            )
            use_wandb = True
        except ImportError:
            print("W&B not installed, running without logging")
            use_wandb = False
    else:
        use_wandb = False
    
    # Initialize the PPO agent
    print(f"Initializing ShieldedPPO for dataset: {args.dataset}")
    agent = ShieldedPPO(
        dataset=args.dataset,
        proj=proj,
        int8=args.int8,
        lr=args.lr,
        kl_beta=args.kl_beta,
        seed=args.seed
    )
    
    # Sample prompts for testing
    prompts = [
        "User likes fantasy novels like LOTR and Harry Potter. Explain why we recommend Mistborn.",
        "User enjoys sci-fi movies with time travel. Explain why we recommend Interstellar.",
        "User likes strategy games with resource management. Explain why we recommend Civilization VI."
    ]
    
    # Use just the first prompt for simplicity
    test_prompt = prompts[0]
    old_logp = torch.zeros(1, device=agent.device)
    
    # Print initial generation
    print("\nInitial generation:")
    initial_text = agent.generate([test_prompt], max_new_tokens=40, do_sample=True, temperature=0.8, top_p=0.9, repetition_penalty=1.1)[0]
    print(f"Prompt: {test_prompt}")
    print(f"Generated: {initial_text}")
    initial_reward = click_reward(initial_text)
    print(f"Reward: {initial_reward}\n")
    
    # Run the PPO loop
    print(f"Running toy PPO loop for {args.steps} steps...")
    start_time = time.time()
    
    all_metrics = []
    for step in range(args.steps):
        # 1. Generate explanation
        with torch.no_grad():
            inputs = agent.tokenizer([test_prompt], return_tensors="pt").to(agent.device)
            gen = agent.model.generate(
                **inputs,
                max_new_tokens=40,
                do_sample=True,
                temperature=0.8,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=agent.tokenizer.eos_token_id,
                # Don't set eos_token_id to allow longer generation
            )
            # Only decode the newly generated tokens (exclude the input prompt)
            new_tokens = gen[0][inputs['input_ids'].shape[1]:]
            generated_text = agent.tokenizer.decode(new_tokens, skip_special_tokens=True)
            text = test_prompt + " " + generated_text  # Full text for reward
            if len(new_tokens) > 1:  # Only print if we actually generated something
                print(f"Generated: '{generated_text[:100]}...' (length: {len(new_tokens)})")
        
        # 2. Compute reward
        r = torch.tensor([click_reward(text)], device=agent.device)
        
        # 3. Run optimization step
        metrics = agent.optimise([test_prompt], r, old_logp)
        all_metrics.append(metrics)
        
        # 4. Log metrics
        if use_wandb:
            wandb.log(metrics, step=step)
        
        # Print progress
        if step % 5 == 0 or step == args.steps - 1:
            print(f"Step {step+1}/{args.steps}: loss={metrics['loss']:.4f}, reward={r.item():.1f}")
    
    end_time = time.time()
    print(f"\nTraining completed in {end_time - start_time:.2f} seconds")
    
    # Print final generation
    print("\nFinal generation:")
    final_text = agent.generate([test_prompt], max_new_tokens=40, do_sample=True, temperature=0.8, top_p=0.9, repetition_penalty=1.1)[0]
    print(f"Prompt: {test_prompt}")
    print(f"Generated: {final_text}")
    final_reward = click_reward(final_text)
    print(f"Reward: {final_reward}\n")
    
    # Save the trained model
    ckpt_path = proj/"checkpoints"/f"recrl_{args.dataset}_toy.pt"
    agent.save_checkpoint(ckpt_path)
    
    # Compute average metrics
    avg_metrics = {k: sum(m[k] for m in all_metrics) / len(all_metrics) for k in all_metrics[0]}
    print("\nAverage metrics:")
    for k, v in avg_metrics.items():
        print(f"{k}: {v:.4f}")
    
    # Check if reward improved
    if final_reward > initial_reward:
        print("\n✅ Success: Reward improved!")
    else:
        print("\n⚠️ Warning: Reward did not improve")
    
    # Close W&B if used
    if use_wandb:
        wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["books", "ml25m", "steam"], default="books")
    parser.add_argument("--int8", action="store_true", help="Use 8-bit quantization")
    parser.add_argument("--steps", type=int, default=30, help="Number of PPO steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--lr", type=float, default=3e-5, help="Learning rate")
    parser.add_argument("--kl_beta", type=float, default=0.05, help="KL penalty coefficient")
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B logging")
    
    args = parser.parse_args()
    main(args)
