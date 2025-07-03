#!/usr/bin/env python
import argparse, pathlib, os, json, yaml
from audit.toxicity import scan_file
from audit.bias import popularity_shift, gender_parity
from audit.privacy import leakage_rate

ROOT = pathlib.Path(os.getenv("PROJ"))
USER_GENDER = ROOT/"data"/"proc"/"ml25m"/"user_gender.csv"   # prepared Section 4

def audit(ds, epoch="final"):
    lora = ROOT/"checkpoints"/f"recrl_{ds}_epoch{epoch}.pt"
    tox  = scan_file(lora, None)
    pop  = popularity_shift(ROOT/f"checkpoints/sasrec_{ds}.pt", lora)
    if ds=="ml25m":
        parity = gender_parity(ROOT/"logs"/f"recs_{ds}.csv", USER_GENDER)
    else:
        parity = {}
    # privacy: sample 1k explanations
    texts = open(ROOT/"logs"/f"expl_{ds}.txt").read().splitlines()[:1000]
    uid   = pd.read_parquet(ROOT/f"data/proc/{ds}/train.parquet")["user"].unique()[:200]
    priv  = leakage_rate(texts, uid)
    report = {"tox": tox,"pop":pop,"parity":parity,"privacy":priv}
    out = ROOT/"docs"/f"safety_report_{ds}.json"
    json.dump(report, open(out,"w"), indent=2)
    print("Report saved", out)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True,
                   choices=["books","ml25m","steam"])
    args = p.parse_args()
    audit(args.dataset)
