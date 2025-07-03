import pandas as pd, numpy as np, json, pathlib, torch

def gini(array):
    """Compute Gini coefficient."""
    array = np.array(array) + 1e-9
    array = np.sort(array)
    n = len(array)
    return (2*np.arange(1,n+1)-n-1).dot(array) / (n*array.sum())

def popularity_shift(ranker_ckpt, lora_ckpt):
    base = torch.load(ranker_ckpt, map_location='cpu')
    items, counts = np.unique(base["item_emb.weight"].argmax(1), return_counts=True)
    gini_base = gini(counts)

    diff = torch.load(lora_ckpt, map_location='cpu')
    shift = diff["base_model.model.lm_head.weight"].abs().sum(1)
    gini_new  = gini(shift.numpy())
    return {"gini_base": float(gini_base), "gini_new": float(gini_new),
            "delta": float(gini_new-gini_base)}

def gender_parity(rec_file, user_gender_csv):
    """MovieLens only."""
    rec = pd.read_csv(rec_file)          # cols: user,item
    dm = pd.read_csv(user_gender_csv)    # cols: user,gender
    merged = rec.merge(dm, on="user")
    clicks = merged.groupby("gender").size()
    rate = clicks / clicks.sum()
    return {"male_rate": rate.get('M',0), "female_rate": rate.get('F',0)}
