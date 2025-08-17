#!/usr/bin/env python
"""
Download GPT-J-6B, attach LoRA adapters, save the *untrained* adapter state.
"""
import argparse, json, hashlib, torch, os, pathlib, time
from load_llm import load_base, add_lora

ROOT = pathlib.Path(os.getenv("PROJ", "."))

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for c in iter(lambda:f.read(1<<20),b""): h.update(c)
    return h.hexdigest()

def main(ds, int8):
    model_name = os.getenv("RECRL_MODEL_NAME", "EleutherAI/gpt-j-6B")
    print(f"Loading {model_name} base model (int8={int8})...")
    tok, base = load_base(int8=int8)
    print("Adding LoRA adapters (r=16)...")
    model = add_lora(base)
    model.print_trainable_parameters()

    out_dir = ROOT/"checkpoints"
    out_dir.mkdir(exist_ok=True)
    out_pt  = out_dir/f"lora_init_{ds}.pt"
    
    print(f"Saving untrained LoRA adapter state to {out_pt}...")
    torch.save(model.state_dict(), out_pt)

    meta = {
        "dataset": ds,
        "model": model_name,
        "int8": int8,
        "lora_r": 16,
        "sha256": sha(out_pt),
        "saved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    }
    meta_path = out_dir/f"lora_init_{ds}_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Metadata saved to {meta_path}")
    print(f"LoRA adapter saved: {out_pt}")
    
    # Print memory usage
    if torch.cuda.is_available():
        vram_used = torch.cuda.max_memory_allocated() / 1e9
        print(f"Peak VRAM usage: {vram_used:.2f} GB")

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=["books","ml25m","steam"])
    p.add_argument("--int8", action="store_true", help="Use 8-bit quantization to reduce VRAM usage")
    args=p.parse_args()
    main(args.dataset, args.int8)
