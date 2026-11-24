"""
Train every model listed in configs/config.yaml, one MLflow run per model.

    python src/train.py --config configs/config.yaml
    python src/train.py --runs naive_bayes_scratch tfidf_logreg      # only some runs

For each run we log to MLflow:
  params    - model hyper-parameters, seed, dataset sizes
  metrics   - per-epoch curves (neural models), accuracy / macro-F1 on validation, test and the
              noisy (typo) test set, calibration error, out-of-scope AUROC / FPR@95TPR for each
              detection method, training time, inference latency, model size
  artifacts - confusion matrix, training curves, out-of-scope score histogram,
              classification report, misclassified examples, model_info.json, the config

The best model (highest VALIDATION macro-F1) is copied to models/best/ for predict.py and the app.
"""

import argparse
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ARTIFACTS_DIR, BEST_DIR, ROOT, load_config, save_json, set_seed, setup_mlflow  # noqa: E402
from data import load_all, noisy_copy  # noqa: E402
from models import build_model  # noqa: E402
from ood import calibrated_probs, evaluate_methods, expected_calibration_error, score  # noqa: E402
from responder import Responder  # noqa: E402
import plots  # noqa: E402

RESULTS_DIR = ROOT / "results"


def flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(flatten(v, f"{prefix}{k}."))
        else:
            out[f"{prefix}{k}"] = v
    return out


def dir_size_mb(path):
    return sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file()) / 1e6


def train_one(run_cfg, cfg, data, labels, responder, mlflow):
    name, kind = run_cfg["name"], run_cfg["type"]
    lab2id = {l: i for i, l in enumerate(labels)}
    tr, va, te, te_noisy = data["train"], data["val"], data["test"], data["test_noisy"]
    y_tr, y_va, y_te = (df["intent"].map(lab2id).values for df in (tr, va, te))
    run_dir = ARTIFACTS_DIR / "runs" / name
    shutil.rmtree(run_dir, ignore_errors=True)
    (run_dir / "plots").mkdir(parents=True)

    print(f"\n=== {name} ({kind}) ===")
    with mlflow.start_run(run_name=name) as run:
        mlflow.set_tags({"model_type": kind,
                         "implemented_from_scratch": str("scratch" in kind).lower(),
                         "dataset": "bitext-customer-support + clinc150-oos(eval only)"})
        mlflow.log_params({"model_type": kind, "seed": cfg["seed"], "n_train": len(tr), "n_val": len(va),
                           "n_test": len(te), "n_classes": len(labels), "noise_rate": cfg["data"]["noise_rate"],
                           **flatten(run_cfg["params"])})

        set_seed(cfg["seed"])
        model = build_model(kind, run_cfg["params"], labels)
        t0 = time.time()
        info = model.fit(tr["text"].tolist(), y_tr, va["text"].tolist(), y_va,
                         log_fn=lambda row, step: mlflow.log_metrics(
                             {f"epoch_{k}": v for k, v in row.items() if k != "epoch"}, step=step))
        train_time = time.time() - t0

        # ---- logits on every evaluation set --------------------------------------
        lg = {"val": model.logits(va["text"]), "test": model.logits(te["text"]),
              "test_noisy": model.logits(te_noisy["text"]),
              "oos_val": model.logits(data["oos_val"]["text"]), "oos_test": model.logits(data["oos_test"]["text"])}
        t1 = time.time()
        for t in te["text"].head(100):
            model.logits([t])
        latency_ms = (time.time() - t1) / min(100, len(te)) * 1000

        m = {"train_time_sec": train_time, "latency_ms_per_query": latency_ms}
        for split, y in (("val", y_va), ("test", y_te), ("test_noisy", y_te)):
            pred = lg[split].argmax(1)
            m[f"{split}_accuracy"] = accuracy_score(y, pred)
            m[f"{split}_macro_f1"] = f1_score(y, pred, average="macro")
        m["robustness_drop_f1"] = m["test_macro_f1"] - m["test_noisy_macro_f1"]

        # ---- out-of-scope detection + calibration ---------------------------------
        ood = evaluate_methods(lg["val"], y_va, lg["oos_val"], lg["test"], lg["oos_test"],
                               keep=cfg["ood"]["keep_in_scope"], method=cfg["ood"]["method"])
        T, chosen = ood["temperature"], ood["chosen"]
        for meth, r in ood["methods"].items():
            m[f"oos_{meth}_val_auroc"] = r["val"]["auroc"]
            m[f"oos_{meth}_test_auroc"] = r["test"]["auroc"]
            m[f"oos_{meth}_test_fpr_at_95tpr"] = r["test"]["fpr_at_95tpr"]
        best_ood = ood["methods"][chosen]
        m.update({"temperature": T, "oos_test_auroc": best_ood["test"]["auroc"],
                  "oos_test_fpr_at_95tpr": best_ood["test"]["fpr_at_95tpr"],
                  "oos_test_recall": best_ood["test"]["oos_recall"],
                  "in_scope_test_accept": best_ood["test"]["in_scope_accept"],
                  "ece_test_raw": expected_calibration_error(calibrated_probs(lg["test"], 1.0), y_te),
                  "ece_test_temperature": expected_calibration_error(calibrated_probs(lg["test"], T), y_te)})
        mlflow.log_param("oos_method_chosen", chosen)

        # ---- save model -----------------------------------------------------------
        model_dir = run_dir / "model"
        model.save(model_dir)
        responder.save(model_dir)
        m["model_size_mb"] = dir_size_mb(model_dir)
        if hasattr(model, "n_parameters"):
            m["n_parameters"] = model.n_parameters()
        mlflow.log_metrics(m)
        model_info = {"kind": kind, "run_name": name, "run_id": run.info.run_id, "labels": labels,
                      "category": responder.category, "temperature": T, "ood_method": chosen,
                      "ood_temperature": best_ood["temperature"], "ood_threshold": best_ood["threshold"],
                      "metrics": m, "ood_details": ood}
        save_json(model_info, model_dir / "model_info.json")

        # ---- plots and reports ------------------------------------------------------
        pred_te = lg["test"].argmax(1)
        P = run_dir / "plots"
        plots.confusion(confusion_matrix(y_te, pred_te, labels=range(len(labels))), labels,
                        P / "confusion_matrix.png", f"{name}: test confusion matrix (row-normalised)")
        t = best_ood["temperature"]
        plots.ood_histogram(score(lg["test"], chosen, t), score(lg["oos_test"], chosen, t), best_ood["threshold"],
                            P / "ood_scores.png", f"{name}: out-of-scope detection ({chosen})")
        if info.get("history"):
            plots.training_curves(info["history"], P / "training_curves.png", name)
            pd.DataFrame(info["history"]).to_csv(run_dir / "history.csv", index=False)
        (run_dir / "classification_report.txt").write_text(
            classification_report(y_te, pred_te, labels=range(len(labels)), target_names=labels, digits=4))
        wrong = pred_te != y_te
        pd.DataFrame({"text": te["text"].values[wrong], "true": np.array(labels)[y_te[wrong]],
                      "predicted": np.array(labels)[pred_te[wrong]]}).to_csv(run_dir / "misclassified_test.csv", index=False)
        save_json({k: v for k, v in model_info.items() if k not in ("labels", "category")}, run_dir / "metrics.json")

        mlflow.log_artifacts(str(P), artifact_path="plots")
        for f in ("classification_report.txt", "misclassified_test.csv", "metrics.json", "history.csv"):
            if (run_dir / f).exists():
                mlflow.log_artifact(str(run_dir / f), artifact_path="reports")
        mlflow.log_artifact(str(model_dir / "model_info.json"), artifact_path="model")
        if cfg["mlflow"].get("log_model_files"):
            mlflow.log_artifacts(str(model_dir), artifact_path="model")
        mlflow.log_artifact(str(ROOT / "configs" / "config.yaml"), artifact_path="config")

        print(f"  val F1 {m['val_macro_f1']:.4f} | test acc {m['test_accuracy']:.4f} | test F1 {m['test_macro_f1']:.4f}"
              f" | noisy F1 {m['test_noisy_macro_f1']:.4f} | OOS AUROC {m['oos_test_auroc']:.4f} ({chosen})"
              f" | {train_time:.1f}s")
        return {"run": name, "type": kind, "run_id": run.info.run_id, **m, "oos_method": chosen}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--runs", nargs="*", help="names of the runs to train (default: all enabled)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["seed"])

    data = load_all(cfg)
    data["test_noisy"] = noisy_copy(data["test"], cfg["data"]["noise_rate"], cfg["seed"])
    labels = sorted(data["train"]["intent"].unique())
    print(f"train {len(data['train']):,} | val {len(data['val']):,} | test {len(data['test']):,} | "
          f"OOS val {len(data['oos_val'])} | OOS test {len(data['oos_test'])} | {len(labels)} intents")
    responder = Responder.build(data["train"], seed=cfg["seed"])
    mlflow = setup_mlflow(cfg)

    runs = [r for r in cfg["runs"] if (r["name"] in args.runs if args.runs else r.get("enabled", True))]
    rows = [train_one(r, cfg, data, labels, responder, mlflow) for r in runs]

    # ---- compare runs and export the best model ------------------------------------
    df = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(exist_ok=True)
    old_csv = RESULTS_DIR / "comparison.csv"
    if args.runs and old_csv.exists():   # retrained only some runs: keep the other runs' results
        old = pd.read_csv(old_csv)
        old = old[~old["run"].isin(df["run"]) & old["run"].apply(lambda r: (ARTIFACTS_DIR / "runs" / r / "model").exists())]
        df = pd.concat([old, df], ignore_index=True)
        rows = df.to_dict("records")
    df.to_csv(RESULTS_DIR / "comparison.csv", index=False)
    plots.comparison(df, RESULTS_DIR / "model_comparison.png")

    # selection uses VALIDATION data only: macro-F1 first, out-of-scope AUROC breaks ties
    best = max(rows, key=lambda r: (round(r["val_macro_f1"], 3), r[f"oos_{r['oos_method']}_val_auroc"]))
    shutil.rmtree(BEST_DIR, ignore_errors=True)
    shutil.copytree(ARTIFACTS_DIR / "runs" / best["run"] / "model", BEST_DIR)
    from mlflow.tracking import MlflowClient
    try:
        MlflowClient().set_tag(best["run_id"], "best_model", "true")
    except Exception as e:   # e.g. mlruns/ was deleted after that run
        print(f"(could not tag the best run in MLflow: {e})")

    cols = ["run", "val_macro_f1", "test_accuracy", "test_macro_f1", "test_noisy_macro_f1",
            "oos_test_auroc", "oos_test_fpr_at_95tpr", "train_time_sec"]
    print("\n" + df[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nBest model (validation macro-F1): {best['run']} -> copied to {BEST_DIR}")
    print("Comparison saved to results/. Open MLflow with:  python src/mlflow_ui.py")


if __name__ == "__main__":
    main()
