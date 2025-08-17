import torch, os, json, math, wandb, random, numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from ppo_core import clipped_surrogate, compute_gae
# Add new imports for RecSim-NG integration
from recsim_ng.core.runtime import runtime
from recsim_ng.core import value

class ShieldedPPO:
    def __init__(self, dataset:str, proj:Path, int8:bool=False,
                 lr=3e-5, kl_beta=0.05, seed=42):
        self.dataset = dataset
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        self.device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # --- Load frozen ranker ---------------------------------------
        ckpt_path = proj/'checkpoints'/f'sasrec_{dataset}.pt'
        self.ranker = torch.load(ckpt_path, map_location='cpu')
        E = self.ranker['item_emb.weight'].float()
        
        # Load projection basis from Section 6
        try:
            basis_path = proj/'checkpoints'/f'basis_{dataset}.pt'
            Q = torch.load(basis_path, map_location='cpu')
        except FileNotFoundError:
            # If basis file doesn't exist, compute it on the fly
            from projection.basis import compute_q
            Q, _ = compute_q(E)              # (d,r)
            
        self.Q = Q.to(self.device)       # used for projection
        # Pre-compute transpose for efficiency
        self.QT = self.Q.t()            # (r,d)

        # --- Load LLM+LoRA -------------------------------------------
        lora_pt = proj/'checkpoints'/f'lora_init_{dataset}.pt'
        from explainer.load_llm import load_base, add_lora
        tok, base = load_base(int8=int8)
        self.tokenizer = tok
        self.model = add_lora(base).to(self.device)
        sd = torch.load(lora_pt, map_location='cpu')
        self.model.load_state_dict(sd, strict=False)

        # --- Optim ----------------------------------------------------
        self.opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.kl_beta = kl_beta
        
        # Log initialization
        print(f"Initialized ShieldedPPO for dataset: {dataset}")
        print(f"Using device: {self.device}")
        print(f"Int8 quantization: {int8}")
        print(f"Learning rate: {lr}")
        print(f"KL beta: {kl_beta}")
        print(f"Projection basis shape: {self.Q.shape}")
        self.model.print_trainable_parameters()

    # ---------------------------------------------------------------
    def projection_step(self):
        """Apply orthogonal projection to gradients to shield the ranker."""
        with torch.no_grad():
            for n, p in self.model.named_parameters():
                if p.grad is None or not p.requires_grad: continue
                grad = p.grad.data
                # Project out components in the Q subspace: g = g - Q(Q^T g)
                g_proj = grad - self.Q @ (self.QT @ grad)
                p.grad.data = g_proj

    # ---------------------------------------------------------------
    def optimise(self, batch_prompts, batch_rewards, old_logprobs):
        """Run a single PPO optimization step."""
        self.model.train()
        
        # Tokenize prompts
        enc = self.tokenizer(batch_prompts, return_tensors="pt",
                             padding=True, truncation=True).to(self.device)
        
        # Forward pass
        out = self.model(**enc, use_cache=False)
        logits = out.logits[:, :-1]  # Shift to align with targets
        
        # Compute log probabilities
        logp = torch.nn.functional.log_softmax(logits, dim=-1)
        new_logprobs = torch.gather(logp, 2, enc.input_ids[:,1:].unsqueeze(-1)).squeeze(-1)
        new_logprobs = new_logprobs.sum(-1)  # Sum over sequence length

        # Compute advantages (since scalar reward, values=0)
        advantages = batch_rewards - 0
        
        # Compute PPO ratio and clipped surrogate objective
        ratio = torch.exp(new_logprobs - old_logprobs)
        loss_pg = -clipped_surrogate(advantages, ratio).mean()
        
        # KL divergence penalty
        loss_kl = self.kl_beta * torch.nn.functional.kl_div(
                      new_logprobs, old_logprobs, log_target=True, reduction='batchmean')
        
        # Total loss
        loss = loss_pg + loss_kl

        # Backward pass with projection
        self.opt.zero_grad()
        loss.backward()
        self.projection_step()  # Apply orthogonal projection to gradients
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)  # Clip gradients
        self.opt.step()
        
        # Return metrics for logging
        return {
            'loss': loss.item(),
            'loss_pg': loss_pg.item(),
            'loss_kl': loss_kl.item(),
            'kl': (new_logprobs-old_logprobs).mean().item(),
            'ratio_mean': ratio.mean().item(),
            'ratio_min': ratio.min().item(),
            'ratio_max': ratio.max().item(),
            'reward_mean': batch_rewards.mean().item()
        }
        
    def generate(self, prompts, max_new_tokens=40, **kwargs):
        """Generate text using the model."""
        self.model.eval()
        with torch.no_grad():
            inputs = self.tokenizer(prompts, return_tensors="pt", padding=True).to(self.device)
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                **kwargs
            )
            return [self.tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    
    # NEW METHOD: Integrate with RecSim-NG simulator
    def rollout_batch(self, batch_size=64, slate_k=5, steps=1):
        """Run simulation steps and collect trajectories with RecSim-NG."""
        # Import the appropriate scenario builder for the dataset
        if not hasattr(self, "sim"):
            if self.dataset == "books":
                from recsim.scenario_books import scenario as build_scenario
            elif self.dataset == "ml25m":
                from recsim.scenario_ml25m import scenario as build_scenario
            elif self.dataset == "steam":
                from recsim.scenario_steam import scenario as build_scenario
            else:
                raise ValueError(f"Unknown dataset: {self.dataset}")
                
            print(f"Building simulator for dataset: {self.dataset}")
            self.sim = build_scenario(batch_size=batch_size, slate_k=slate_k)
            self.runner = runtime.Runtime(self.sim, seed=42)  # Fixed seed for reproducibility
            
            # Initialize state with random user history
            item_count = self.ranker["item_emb.weight"].shape[0]
            init_state = {"user": {"user_bias": np.random.normal(0, 0.3, batch_size)},
                          "last_item": np.random.randint(0, item_count, batch_size)}
            self.runner.init_state(init_state)
        
        trajectories = []
        
        for _ in range(steps):
            # 1. Get current state
            state = self.runner.get_state()
            user_biases = state["user"]["user_bias"]
            user_histories = state["last_item"]
            
            # 2. Sample slate of items for each user
            item_count = self.ranker["item_emb.weight"].shape[0]
            slates = np.zeros((batch_size, slate_k), dtype=np.int32)
            
            # For each user, sample top-k items based on ranker scores
            item_emb = self.ranker["item_emb.weight"].cpu().numpy()
            for i in range(batch_size):
                # Use the last item in user history to get similar items
                last_item_idx = user_histories[i]
                # Compute similarity scores
                similarity = item_emb @ item_emb[last_item_idx].T
                # Get top-k items (excluding the last item itself)
                top_items = np.argsort(similarity)[::-1][:slate_k+1]
                if last_item_idx in top_items:
                    top_items = np.delete(top_items, np.where(top_items == last_item_idx))
                slates[i] = top_items[:slate_k]
            
            # 3. Build prompts for each user-item pair
            prompts = []
            for i in range(batch_size):
                hist_item = user_histories[i]
                rec_item = slates[i][0]  # Use first item in slate
                prompt = f"User previously engaged with item {hist_item}. Explain why we recommend item {rec_item}."
                prompts.append(prompt)
            
            # 4. Generate explanations
            explanations = self.generate(prompts, max_new_tokens=40,
                                        do_sample=True, temperature=0.7)
            
            # 5. Compute explanation quality based on length
            expl_qualities = np.zeros((batch_size, slate_k), dtype=np.float32)
            for i in range(batch_size):
                # For simplicity, measure quality as normalized token length
                tokens = len(explanations[i].split())
                # Normalize to [0,1] with maximum at 40 tokens
                quality = min(tokens / 40.0, 1.0)
                # Apply to all items in the slate (in practice, would be item-specific)
                expl_qualities[i, :] = quality
            
            # 6. Update RecSim-NG state with explanation quality and get clicks
            next_state = self.runner.step({
                "slate": slates,
                "expl_qual": expl_qualities
            })
            
            # Extract click rewards
            clicks = next_state["user"]["clicked"].numpy()
            
            # Store trajectories
            for i in range(batch_size):
                trajectories.append({
                    "user_id": i,
                    "user_bias": user_biases[i],
                    "history": user_histories[i],
                    "slate": slates[i].tolist(),
                    "prompt": prompts[i],
                    "explanation": explanations[i],
                    "expl_quality": expl_qualities[i, 0],
                    "reward": clicks[i],
                })
        
        return trajectories
    
    def save_checkpoint(self, path, epoch=None):
        """Save LoRA weights and optimizer state."""
        checkpoint = {
            'model': self.model.state_dict(),
            'optimizer': self.opt.state_dict(),
            'dataset': self.dataset,
            'epoch': epoch,
            'timestamp': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'
        }
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")
        
    def load_checkpoint(self, path):
        """Load LoRA weights and optimizer state."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model'], strict=False)
        self.opt.load_state_dict(checkpoint['optimizer'])
        print(f"Loaded checkpoint from {path} (epoch {checkpoint.get('epoch', 'unknown')})")
        return checkpoint
