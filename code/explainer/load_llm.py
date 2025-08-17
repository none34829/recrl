"""Utility to load GPT-J-6B with optional bitsandbytes 8-bit + LoRA adapters."""

from pathlib import Path
import torch, os
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model

HF_CACHE = Path(os.getenv("HF_HOME", "/workspace/.cache/huggingface"))

# >>> ADD near the top of load_llm.py <<<
DEFAULT_MODEL = "EleutherAI/gpt-j-6B"  # original default for big runs
MODEL_NAME = os.getenv("RECRL_MODEL_NAME", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")

USE_4BIT = os.getenv("RECRL_4BIT", "1") == "1"
MAX_GPU_MEM = os.getenv("RECRL_MAX_GPU_MEM", "5GiB")  # for your 6 GB 4050
CPU_MEM = os.getenv("RECRL_CPU_MEM", "48GiB")         # offload target

bnb_config = None
quant_kwargs = {}
if USE_4BIT:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_compute_dtype="bfloat16",
        bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4",
    )
    quant_kwargs = {
        "quantization_config": bnb_config,
        "device_map": "auto",
        "max_memory": {0: MAX_GPU_MEM, "cpu": CPU_MEM},
        "torch_dtype": torch.bfloat16,
    }
else:
    quant_kwargs = {"device_map": "auto"}

def load_base(model_name=MODEL_NAME,
              int8=False, device="cuda"):
    kwargs = dict(torch_dtype="auto", device_map="auto",
                  cache_dir=HF_CACHE)
    
    # Use the new quantization settings
    if USE_4BIT:
        kwargs.update(quant_kwargs)
    elif int8:
        kwargs.update(load_in_8bit=True)
    
    tok  = AutoTokenizer.from_pretrained(model_name, use_fast=True, **kwargs)
    mod  = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    
    # (optional but helpful on 6GB) - DISABLED for gradient flow testing
    # if hasattr(mod, 'gradient_checkpointing_enable'):
    #     mod.gradient_checkpointing_enable()
    
    return tok, mod

def add_lora(model, r=16, alpha=16):
    cfg = LoraConfig(
        r=r, lora_alpha=alpha, target_modules=["q_proj","v_proj",
                                               "k_proj","o_proj"],
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
    return get_peft_model(model, cfg)
