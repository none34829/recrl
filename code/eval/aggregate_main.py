#!/usr/bin/env python
"""
Harvests metric JSONs from checkpoints/, docs/, and produces:
  • docs/main_results.{csv,tex}
  • docs/fig_ctr_curve.png (CTR vs epoch)
"""
import pathlib, json, glob, csv, os
import pandas as pd, matplotlib.pyplot as plt

ROOT = pathlib.Path(os.getenv("PROJ", "."))
rows = []
for ds in ["books", "ml25m", "steam"]:
    # 1. offline NDCG
    ndcg = json.load(open(ROOT/f"checkpoints/sasrec_{ds}_stats.json"))["ndcg10"]
    # 2. RLHF metrics
    meta = json.load(open(ROOT/f"checkpoints/recrl_{ds}_epoch7_meta.json"))
    ctr, kl = meta["ctr"], meta["kl"]
    # 3. safety
    tox = json.load(open(ROOT/f"docs/safety_report_{ds}.json"))["tox"]["mean"]
    rows.append(dict(dataset=ds, ndcg10_off=ndcg,
                     ctr_sim=ctr, kl_final=kl, tox=tox))

df = pd.DataFrame(rows)
csv_path = ROOT/"docs"/"main_results.csv"
df.to_csv(csv_path, index=False)

# LaTeX table
tex_path = ROOT/"docs"/"main_results.tex"
with open(tex_path, "w") as tex:
    tex.write("\\begin{tabular}{lrrrr}\n\\toprule\n")
    tex.write("Dataset & NDCG$_{10}$ & CTR$_\\text{sim}$ "
              "& KL & Toxicity\\\\\n\\midrule\n")
    for r in rows:
        tex.write(f"{r['dataset']} & {r['ndcg10_off']:.3f}"
                  f" & {r['ctr_sim']:.3f} & {r['kl_final']:.3f}"
                  f" & {r['tox']:.3f}\\\\\n")
    tex.write("\\bottomrule\n\\end{tabular}\n")

# CTR curve – gather per-epoch JSONs
plt.figure(figsize=(4,2.5))
for ds,c in zip(["books","ml25m","steam"],["C0","C1","C2"]):
    ctrs=[]
    for ep in range(8):
        meta = json.load(open(
            ROOT/f"checkpoints/recrl_{ds}_epoch{ep}_meta.json"))
        ctrs.append(meta["ctr"])
    plt.plot(range(8), ctrs, label=ds, color=c)
plt.xlabel("epoch"); plt.ylabel("CTR (sim)"); plt.legend()
plt.tight_layout()
fig_path = ROOT/"docs"/"fig_ctr_curve.png"
plt.savefig(fig_path, dpi=300)

# Manifest updates
def digest(p):
    import hashlib
    h = hashlib.sha256()
    h.update(open(p,"rb").read())
    return h.hexdigest()

manifest = ROOT/"data/_checksums/results_manifest.jsonl"
os.makedirs(os.path.dirname(manifest), exist_ok=True)

for p in [csv_path, tex_path, fig_path]:
    with open(manifest,"a") as mf:
        mf.write(json.dumps({"type":"result",
                             "file":p.name,
                             "sha256":digest(p),
                             "date":pd.Timestamp.now().isoformat()})+"\n")
print("Wrote", csv_path, tex_path, fig_path)
