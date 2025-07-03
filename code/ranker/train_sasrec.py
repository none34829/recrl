import os, json, random, argparse, torch, numpy as np, pandas as pd, math
from pathlib import Path
from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader
from models.sasrec import SASRec

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False

class SeqDataset(Dataset):
    def __init__(self, parquet, max_len=50):
        df = pd.read_parquet(parquet)
        grp = df.groupby("user")
        self.seqs = [g.sort_values("ts")["item"].tolist() for _,g in grp]
        self.max_len=max_len
    def __len__(self): return len(self.seqs)
    def __getitem__(self, idx):
        seq = self.seqs[idx]
        tgt = seq[-1]
        seq = seq[-(self.max_len+1):-1]   # last max_len items
        seq = [0]*(self.max_len-len(seq)) + seq
        return torch.tensor(seq, dtype=torch.long), torch.tensor(tgt)

def metrics(model, loader, device):
    model.eval(); hits=ndcg=tot=0
    with torch.no_grad():
        for seq,tgt in loader:
            seq,tgt = seq.to(device), tgt.to(device)
            logits  = model(seq)
            _, idx  = logits.topk(10)          # (B,10)
            for i in range(len(tgt)):
                tot += 1
                if tgt[i] in idx[i]: hits+=1
                rank = (idx[i]==tgt[i]).nonzero(as_tuple=False)
                if rank.numel(): ndcg += 1/math.log2(rank.item()+2)
    return hits/tot, ndcg/tot

def main(args):
    seed_all(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_ds = SeqDataset(Path(args.proc_dir)/"train.parquet")
    val_ds   = SeqDataset(Path(args.proc_dir)/"valid.parquet")
    
    # Find the maximum item ID in the training set
    max_item = max(max(seq) for seq in train_ds.seqs)
    n_items  = max_item + 1
    print(f"Dataset: {args.dataset}, Items: {n_items:,}")
    
    model    = SASRec(n_items).to(device)
    opt      = torch.optim.Adam(model.parameters(), 1e-3)

    train_ld = DataLoader(train_ds, 256, shuffle=True, num_workers=4)
    val_ld   = DataLoader(val_ds,   256, shuffle=False, num_workers=4)

    # Create checkpoints directory if it doesn't exist
    Path(args.ckpt).parent.mkdir(parents=True, exist_ok=True)
    
    # Initialize W&B if available and not disabled
    try:
        import wandb
        if os.environ.get("WANDB_DISABLED", "true").lower() != "true":
            wandb.init(
                project="recrl_ranker",
                name=f"sasrec_{args.dataset}_seed{args.seed}",
                config={
                    "dataset": args.dataset,
                    "seed": args.seed,
                    "n_items": n_items,
                    "hidden": 128,
                    "n_layers": 2,
                    "max_len": 50,
                    "n_heads": 2,
                    "dropout": 0.2,
                    "batch_size": 256,
                    "lr": 1e-3
                }
            )
            use_wandb = True
        else:
            use_wandb = False
    except ImportError:
        use_wandb = False

    best_ndcg = 0
    best_epoch = -1
    best_hr = 0
    
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        batch_count = 0
        
        for seq,tgt in tqdm(train_ld, desc=f"Epoch {epoch+1}/{args.epochs}"):
            seq,tgt = seq.to(device), tgt.to(device)
            logits  = model(seq)
            loss    = torch.nn.functional.cross_entropy(logits, tgt)
            opt.zero_grad(); loss.backward(); opt.step()
            
            epoch_loss += loss.item()
            batch_count += 1
        
        avg_loss = epoch_loss / batch_count
        hit, ndcg = metrics(model, val_ld, device)
        print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, HR@10={hit:.4f}, NDCG@10={ndcg:.4f}")
        
        # Log to W&B if available
        if use_wandb:
            wandb.log({
                "epoch": epoch,
                "train_loss": avg_loss,
                "val_hr10": hit,
                "val_ndcg10": ndcg
            })
        
        # Save checkpoint if NDCG improved
        if ndcg > best_ndcg:
            best_ndcg = ndcg
            best_hr = hit
            best_epoch = epoch
            
            # Save model checkpoint
            torch.save(model.state_dict(), args.ckpt)
            
            # Save stats
            stats = {
                "dataset": args.dataset,
                "seed": args.seed,
                "n_items": int(n_items),
                "hr10": float(hit),
                "ndcg10": float(ndcg),
                "epoch": epoch,
                "hidden": 128,
                "n_layers": 2,
                "max_len": 50,
                "n_heads": 2,
                "dropout": 0.2
            }
            
            with open(args.stat, "w") as f:
                json.dump(stats, f, indent=2)
    
    print(f"\nTraining complete!")
    print(f"Best validation metrics (epoch {best_epoch}):")
    print(f"  HR@10:   {best_hr:.4f}")
    print(f"  NDCG@10: {best_ndcg:.4f}")
    
    # Compute SHA-256 of checkpoint
    import hashlib
    with open(args.ckpt, "rb") as f:
        ckpt_hash = hashlib.sha256(f.read()).hexdigest()
    
    print(f"Checkpoint SHA-256: {ckpt_hash}")
    
    # Add to manifest
    manifest_path = Path(os.getenv("PROJ", ".")) / "data" / "_checksums" / "ranker_manifest.jsonl"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    
    manifest_entry = {
        "dataset": args.dataset,
        "sha256": ckpt_hash,
        "timestamp": pd.Timestamp.now().isoformat()
    }
    
    with open(manifest_path, "a") as f:
        f.write(json.dumps(manifest_entry) + "\n")
    
    # Close W&B if used
    if use_wandb:
        wandb.finish()

if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=["books","ml25m","steam"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=10)
    args = p.parse_args()

    proj = Path(os.getenv("PROJ", "."))
    args.proc_dir = proj/"data"/"proc"/args.dataset
    args.ckpt     = proj/"checkpoints"/f"sasrec_{args.dataset}.pt"
    args.stat     = proj/"checkpoints"/f"sasrec_{args.dataset}_stats.json"
    main(args)
