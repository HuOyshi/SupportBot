"""
Threshold analysis for out-of-scope detection.

    python src/threshold_check.py

For every trained model and two scoring methods it compares threshold rules chosen on
VALIDATION data only (keep 95% / 99% / 99.5% / 99.9% of in-scope messages, and Youden's J,
which balances in-scope acceptance against out-of-scope rejection on the validation OOS set).
It also shows how each rule handles the hand-written messages in samples/sample_queries.csv,
which are phrased more like real customers than the Bitext data.

Writes results/threshold_analysis.csv and results/sample_predictions_by_model.csv
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ARTIFACTS_DIR, ROOT, load_config  # noqa: E402
from data import load_all  # noqa: E402
from models import load_model  # noqa: E402
from ood import score, threshold_for  # noqa: E402

RUNS = ["tfidf_logreg", "transformer_scratch_subword", "transformer_scratch_word", "naive_bayes_scratch"]
OOS_SAMPLES = {"what's the weather like tomorrow", "tell me a joke",
               "who won the football match yesterday", "book me a flight to Paris"}
KEEPS = [0.95, 0.99, 0.995, 0.999]


def youden_threshold(in_scores, oos_scores):
    """Threshold that maximises (in-scope accepted) + (out-of-scope rejected) on validation."""
    cand = np.unique(np.r_[in_scores, oos_scores])
    acc = (in_scores[None, :] >= cand[:, None]).mean(1)
    rej = (oos_scores[None, :] < cand[:, None]).mean(1)
    return float(cand[int(np.argmax(acc + rej))])


def main():
    cfg = load_config()
    data = load_all(cfg)
    samples = pd.read_csv(ROOT / "samples" / "sample_queries.csv")["text"].astype(str).tolist()
    is_oos = np.array([t in OOS_SAMPLES for t in samples])
    rows, per_sample = [], pd.DataFrame({"text": samples, "is_out_of_scope": is_oos})

    for run in RUNS:
        d = ARTIFACTS_DIR / "runs" / run / "model"
        if not (d / "model_info.json").exists():
            continue
        model, info = load_model(d)
        labels, T = info["labels"], info["temperature"]
        lv, lo, ls = (model.logits(x) for x in (data["val"]["text"], data["oos_val"]["text"], samples))
        pred = ls.argmax(1)
        per_sample[f"{run}_intent"] = [labels[i] for i in pred]
        for method in ("energy", "temp_msp"):
            sv, so, ss = (score(x, method, T) for x in (lv, lo, ls))
            per_sample[f"{run}_{method}_score"] = np.round(ss, 3)
            rules = [(f"keep {k}", threshold_for(sv, k)) for k in KEEPS] + [("youden", youden_threshold(sv, so))]
            for rule, thr in rules:
                rows.append({
                    "model": run, "method": method, "rule": rule, "threshold": round(thr, 4),
                    "val_in_accept": round(float((sv >= thr).mean()), 4),
                    "val_oos_reject": round(float((so < thr).mean()), 4),
                    "samples_in_accepted": f"{int((ss[~is_oos] >= thr).sum())}/{int((~is_oos).sum())}",
                    "samples_oos_rejected": f"{int((ss[is_oos] < thr).sum())}/{int(is_oos.sum())}",
                })

    df = pd.DataFrame(rows)
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    df.to_csv(out / "threshold_analysis.csv", index=False)
    per_sample.to_csv(out / "sample_predictions_by_model.csv", index=False)
    pd.set_option("display.width", 200)
    print(df.to_string(index=False))
    print("\nSample intents by model:")
    print(per_sample[["text"] + [c for c in per_sample.columns if c.endswith("_intent")]].to_string(index=False))


if __name__ == "__main__":
    main()
