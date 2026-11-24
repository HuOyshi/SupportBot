"""
Evaluate the exported best model (models/best/) on the held-out data.

    python src/evaluate.py

Writes to results/:
  evaluation.md / evaluation.json   metric table for the report
  best_confusion_matrix.png         test confusion matrix
  best_ood_scores.png               in-scope vs out-of-scope score distribution
  per_intent_report.csv             precision / recall / F1 per intent
  sample_predictions.csv            full pipeline output on samples/sample_queries.csv
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import BEST_DIR, ROOT, load_config, save_json  # noqa: E402
from data import load_all, noisy_copy  # noqa: E402
from ood import calibrated_probs, expected_calibration_error, ood_metrics, score  # noqa: E402
from predict import SupportBot  # noqa: E402
import plots  # noqa: E402

RESULTS = ROOT / "results"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--model-dir", default=str(BEST_DIR))
    args = ap.parse_args()
    cfg = load_config(args.config)
    RESULTS.mkdir(exist_ok=True)

    bot = SupportBot(args.model_dir, args.config)
    info, labels = bot.info, bot.labels
    lab2id = {l: i for i, l in enumerate(labels)}
    data = load_all(cfg)
    test = data["test"]
    noisy = noisy_copy(test, cfg["data"]["noise_rate"], cfg["seed"])
    y = test["intent"].map(lab2id).values

    lg_test, lg_noisy, lg_oos = (bot.model.logits(d["text"]) for d in (test, noisy, data["oos_test"]))
    pred, pred_noisy = lg_test.argmax(1), lg_noisy.argmax(1)
    T, meth, t_ood, thr = info["temperature"], info["ood_method"], info["ood_temperature"], info["ood_threshold"]
    s_in, s_oos = score(lg_test, meth, t_ood), score(lg_oos, meth, t_ood)
    om = ood_metrics(s_in, s_oos, thr)

    # end-to-end accuracy: an in-scope message counts as correct only if it is accepted AND classified correctly
    accepted = s_in >= thr
    metrics = {
        "model": info["run_name"], "mlflow_run_id": info["run_id"],
        "test_accuracy": accuracy_score(y, pred), "test_macro_f1": f1_score(y, pred, average="macro"),
        "test_weighted_f1": f1_score(y, pred, average="weighted"),
        "noisy_test_accuracy": accuracy_score(y, pred_noisy),
        "noisy_test_macro_f1": f1_score(y, pred_noisy, average="macro"),
        "ece_raw": expected_calibration_error(calibrated_probs(lg_test, 1.0), y),
        "ece_temperature_scaled": expected_calibration_error(calibrated_probs(lg_test, T), y),
        "temperature": T, "ood_method": meth,
        "oos_auroc": om["auroc"], "oos_fpr_at_95tpr": om["fpr_at_95tpr"],
        "oos_recall_at_threshold": om["oos_recall"], "in_scope_accept_rate": om["in_scope_accept"],
        "end_to_end_in_scope_accuracy": float((accepted & (pred == y)).mean()),
        "n_test": int(len(test)), "n_oos_test": int(len(data["oos_test"])),
    }

    rep = classification_report(y, pred, labels=range(len(labels)), target_names=labels, output_dict=True)
    pd.DataFrame(rep).T.iloc[:len(labels)].round(4).to_csv(RESULTS / "per_intent_report.csv")
    plots.confusion(confusion_matrix(y, pred, labels=range(len(labels))), labels,
                    RESULTS / "best_confusion_matrix.png", f"{info['run_name']}: test confusion matrix")
    plots.ood_histogram(s_in, s_oos, thr, RESULTS / "best_ood_scores.png",
                        f"{info['run_name']}: out-of-scope detection ({meth})")

    sample_file = ROOT / "samples" / "sample_queries.csv"
    if sample_file.exists():
        res = pd.DataFrame(bot.predict(pd.read_csv(sample_file)["text"].tolist()))
        res["top_intents"] = res["top_intents"].apply(lambda t: "; ".join(f"{i}:{p}" for i, p in t))
        res.to_csv(RESULTS / "sample_predictions.csv", index=False)
        metrics["sample_escalation_rate"] = float(res["escalate"].mean())

    save_json(metrics, RESULTS / "evaluation.json")
    rows = [("Test accuracy", "test_accuracy"), ("Test macro-F1", "test_macro_f1"),
            ("Test weighted-F1", "test_weighted_f1"), ("Noisy test accuracy (typos)", "noisy_test_accuracy"),
            ("Noisy test macro-F1 (typos)", "noisy_test_macro_f1"), ("ECE before temperature scaling", "ece_raw"),
            ("ECE after temperature scaling", "ece_temperature_scaled"), ("Out-of-scope AUROC", "oos_auroc"),
            ("Out-of-scope FPR@95%TPR", "oos_fpr_at_95tpr"), ("Out-of-scope recall at threshold", "oos_recall_at_threshold"),
            ("In-scope acceptance rate", "in_scope_accept_rate"),
            ("End-to-end in-scope accuracy", "end_to_end_in_scope_accuracy")]
    md = [f"# Evaluation of the best model: `{info['run_name']}`", "",
          f"MLflow run id: `{info['run_id']}`  |  OOS method: `{meth}`  |  temperature T = {T:.3f}", "",
          "| Metric | Value |", "|---|---|"] + [f"| {n} | {metrics[k]:.4f} |" for n, k in rows]
    comp = RESULTS / "comparison.csv"
    if comp.exists():
        c = pd.read_csv(comp)
        cols = ["run", "val_macro_f1", "test_accuracy", "test_macro_f1", "test_noisy_macro_f1",
                "oos_test_auroc", "oos_test_fpr_at_95tpr", "train_time_sec", "latency_ms_per_query"]
        cols = [x for x in cols if x in c.columns]
        md += ["", "## All runs (from train.py)", "", "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
        md += ["| " + " | ".join(str(r[x]) if x == "run" else f"{r[x]:.4f}" for x in cols) + " |" for _, r in c.iterrows()]
        ood_cols = [x for x in c.columns if x.startswith("oos_") and x.endswith("_test_auroc") and x != "oos_test_auroc"]
        if ood_cols:
            md += ["", "## Out-of-scope AUROC by detection method (test)", "",
                   "| run | " + " | ".join(x[4:-11] for x in ood_cols) + " |", "|---|" + "---|" * len(ood_cols)]
            md += [f"| {r['run']} | " + " | ".join(f"{r[x]:.4f}" for x in ood_cols) + " |" for _, r in c.iterrows()]
    (RESULTS / "evaluation.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    print(f"\nSaved to {RESULTS}")


if __name__ == "__main__":
    main()
