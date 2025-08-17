import tensorflow as tf, recsim_ng as rs
# from recsim_ng.core import pipeline  # Commented out - not available in this version
from recsim_ng.core.network import FunctionalContext
from recsim.base_user import SimpleUser
import torch, os, pathlib, numpy as np, pandas as pd
# ------------------------------------------------------------------
PROJ = pathlib.Path(os.getenv("PROJ", "."))
PROC = PROJ/"data"/"proc"/"books"
RANKER = torch.load(PROJ/"checkpoints"/"sasrec_books.pt", map_location="cpu")

# Precompute per-item ranker logits for speed
item_emb = RANKER["item_emb.weight"]
scores_fn = lambda hist, item_idx: (item_emb[item_idx] @ item_emb[hist[-1]]).item()

def scenario(batch_size=64, slate_k=5):
    # Simple fallback scenario without RecSim-NG pipeline
    # This provides basic functionality for the training loop
    ctx = FunctionalContext()
    user = SimpleUser()
    ctx.add_entity(user)
    # docs are static items; sample slate externally
    def _response(prev_state, slate):
        hist = tf.cast(prev_state["last_item"], tf.int32)
        score = tf.constant([scores_fn(hist.numpy(), i)
                             for i in slate.numpy()], dtype=tf.float32)
        expl_q = tf.ones_like(score)  # filled later
        return user.next_response(prev_state, slate, score, expl_q)
    ctx.add_response(_response)
    # Return a simple object instead of pipeline.Pipeline(ctx)
    return ctx
