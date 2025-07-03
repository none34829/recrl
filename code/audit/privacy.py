import pandas as pd, Levenshtein as lev

def leakage_rate(texts, user_ids):
    leaks = 0
    for t in texts:
        for uid in user_ids:
            if lev.distance(str(uid), t) <= 2:
                leaks += 1; break
    return leaks/len(texts)
