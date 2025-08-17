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
from reward import improved_click_reward

def compute_logprobs(model, tokenizer, prompt, max_new_tokens=80, **kwargs):
    """Compute log probabilities for generated tokens."""
    model.eval()
    with torch.no_grad():
        # Tokenize the prompt
        inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
        
        # Generate tokens
        gen = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.9,
            top_p=0.95,
            repetition_penalty=1.05,
            pad_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=True  # This gives us the logits for each generated token
        )
        
        # Extract the generated tokens (exclude the input prompt)
        new_tokens = gen.sequences[0][inputs['input_ids'].shape[1]:]
        
        # Compute log probabilities for the generated tokens
        logprobs = []
        for i, token_id in enumerate(new_tokens):
            if i < len(gen.scores):
                logits = gen.scores[i][0]  # Get logits for this token
                log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
                logprob = log_probs[token_id].item()
                logprobs.append(logprob)
        
        # Decode the generated text
        generated_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        
        return generated_text, torch.tensor(logprobs, device=model.device)

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
    
    # Better prompts that are more likely to generate longer responses
    prompts = [
        "Write a detailed explanation of why Mistborn would appeal to fans of Lord of the Rings and Harry Potter. Include specific elements like magic systems, world-building, and character development.",
        "Explain why we recommend the Mistborn series to fantasy readers. Describe the unique aspects of Brandon Sanderson's writing style and how it compares to other fantasy authors.",
        "Give a comprehensive recommendation for Mistborn. Discuss the magic system, plot complexity, and why it's perfect for readers who enjoyed epic fantasy series."
    ]
    
    # Use just the first prompt for simplicity
    test_prompt = prompts[0]
    
    # Print initial generation
    print("\nInitial generation:")
    initial_text, initial_logprobs = compute_logprobs(agent.model, agent.tokenizer, test_prompt)
    print(f"Prompt: {test_prompt}")
    print(f"Generated: {initial_text}")
    initial_reward = improved_click_reward(initial_text)
    print(f"Reward: {initial_reward:.3f}\n")
    
    # Run the PPO loop
    print(f"Running toy PPO loop for {args.steps} steps...")
    start_time = time.time()
    
    all_metrics = []
    old_logprobs = initial_logprobs  # Start with the initial generation's logprobs
    
    for step in range(args.steps):
        # 1. Generate explanation and compute log probabilities
        generated_text, new_logprobs = compute_logprobs(agent.model, agent.tokenizer, test_prompt)
        text = test_prompt + " " + generated_text  # Full text for reward
        
        if len(new_logprobs) > 1:  # Only print if we actually generated something
            print(f"Generated: '{generated_text[:150]}...' (length: {len(new_logprobs)})")
        
        # 2. Compute reward using improved function
        r = torch.tensor([improved_click_reward(text)], device=agent.device)
        
        # 3. Run optimization step with proper log probabilities
        metrics = agent.optimise([test_prompt], r, old_logprobs)
        all_metrics.append(metrics)
        
        # 4. Log metrics
        if use_wandb:
            wandb.log(metrics, step=step)
        
        # Print progress
        if step % 5 == 0 or step == args.steps - 1:
            print(f"Step {step+1}/{args.steps}: loss={metrics['loss']:.4f}, reward={r.item():.3f}")
        
        # Store current logprobs for next iteration
        old_logprobs = new_logprobs.detach()
    
    end_time = time.time()
    print(f"\nTraining completed in {end_time - start_time:.2f} seconds")
    
    # Print final generation
    print("\nFinal generation:")
    final_text, _ = compute_logprobs(agent.model, agent.tokenizer, test_prompt)
    print(f"Prompt: {test_prompt}")
    print(f"Generated: {final_text}")
    final_reward = improved_click_reward(final_text)
    print(f"Reward: {final_reward:.3f}\n")
    
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
        print(f"\n✅ Success: Reward improved from {initial_reward:.3f} to {final_reward:.3f}!")
    else:
        print(f"\n⚠️ Warning: Reward did not improve (initial: {initial_reward:.3f}, final: {final_reward:.3f})")
    
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
