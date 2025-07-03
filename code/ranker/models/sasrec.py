import torch, math, torch.nn as nn

class SASRec(nn.Module):
    def __init__(self, n_items:int, hidden:int=128, n_layers:int=2,
                 max_len:int=50, n_heads:int=2, dropout:float=0.2):
        super().__init__()
        self.item_emb = nn.Embedding(n_items+1, hidden, padding_idx=0)
        self.pos_emb  = nn.Embedding(max_len, hidden)
        self.layers   = nn.ModuleList([
            nn.TransformerEncoderLayer(hidden, n_heads, hidden*4,
                                       dropout, batch_first=True)
            for _ in range(n_layers)
        ])
        self.dropout  = nn.Dropout(dropout)
        self.max_len  = max_len
        nn.init.normal_(self.item_emb.weight, std=0.02)

    def forward(self, seq):                          # (B,L)
        b, l = seq.size()
        pos = torch.arange(l, device=seq.device).unsqueeze(0)
        x = self.item_emb(seq) + self.pos_emb(pos)
        x = self.dropout(x)
        attn_mask = ~seq.bool().unsqueeze(1).expand(-1,l,-1)   # padding mask
        for layer in self.layers:
            x = layer(x, src_key_padding_mask=~seq.bool())
        out = x[:, -1, :]                               # last position
        logits = out @ self.item_emb.weight.T           # B × n_items
        return logits
