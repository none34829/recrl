"""Utility to load GPT-J-6B with optional bitsandbytes 8-bit + LoRA adapters."""

from pathlib import Path
import torch, os
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

HF_CACHE = Path(os.getenv("HF_HOME", "/workspace/.cache/huggingface"))

def load_base(model_name="EleutherAI/gpt-j-6B",
              int8=False, device="cuda"):
    kwargs = dict(torch_dtype="auto", device_map="auto",
                  cache_dir=HF_CACHE)
    if int8:
        kwargs.update(load_in_8bit=True)
    tok  = AutoTokenizer.from_pretrained(model_name, **kwargs)
    mod  = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    return tok, mod

def add_lora(model, r=16, alpha=16):
    cfg = LoraConfig(
        r=r, lora_alpha=alpha, target_modules=["q_proj","v_proj",
                                               "k_proj","o_proj"],
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
    return get_peft_model(model, cfg)
