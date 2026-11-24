"""
Out-of-scope (OOS) detection: decide when a message is NOT a customer-support request
("what's the weather tomorrow?") so the bot does not answer it with a random intent.

Every method turns the classifier's logits into an "in-scope score" (higher = more in-scope):

  msp       Maximum Softmax Probability - the confidence of the top intent (Hendrycks & Gimpel, 2017)
  temp_msp  MSP after temperature scaling: logits / T, with T fitted on validation data
            to minimise negative log-likelihood (Guo et al., 2017). Also gives calibrated confidence.
  energy    Negative energy  T * logsumexp(logits / T)  (Liu et al., 2020). Uses all logits,
            not just the largest one.

A message is flagged out-of-scope when its score is below a threshold chosen so that
`keep_in_scope` (e.g. 95%) of in-scope VALIDATION messages are still accepted.
"""

import numpy as np
from scipy.special import logsumexp, log_softmax, softmax
from sklearn.metrics import roc_auc_score

METHODS = ("msp", "temp_msp", "energy")


def fit_temperature(logits, y):
    """Grid search for the temperature T that minimises validation NLL."""
    y = np.asarray(y)
    grid = np.exp(np.linspace(np.log(0.1), np.log(100.0), 300))   # T < 0.1 only makes the model more over-confident
    nll = [-log_softmax(logits / t, axis=1)[np.arange(len(y)), y].mean() for t in grid]
    return float(grid[int(np.argmin(nll))])


def calibrated_probs(logits, temperature=1.0):
    return softmax(np.asarray(logits, dtype=np.float64) / temperature, axis=1)


def score(logits, method, temperature=1.0):
    logits = np.asarray(logits, dtype=np.float64)
    if method == "msp":
        return softmax(logits, axis=1).max(1)
    if method == "temp_msp":
        return softmax(logits / temperature, axis=1).max(1)
    if method == "energy":
        return temperature * logsumexp(logits / temperature, axis=1)
    raise ValueError(f"unknown OOS method {method}")


def threshold_for(in_scope_scores, keep=0.95):
    """Score below which a message is rejected; keeps `keep` of in-scope messages."""
    return float(np.quantile(in_scope_scores, 1.0 - keep))


def ood_metrics(in_scores, oos_scores, threshold):
    """AUROC, FPR@95%TPR and the effect of the chosen threshold."""
    y = np.r_[np.ones(len(in_scores)), np.zeros(len(oos_scores))]
    s = np.r_[in_scores, oos_scores]
    t95 = np.quantile(in_scores, 0.05)    # threshold that keeps 95% of in-scope messages
    return {
        "auroc": float(roc_auc_score(y, s)),
        "fpr_at_95tpr": float((oos_scores >= t95).mean()),        # OOS wrongly accepted (lower = better)
        "oos_recall": float((oos_scores < threshold).mean()),     # OOS correctly rejected
        "in_scope_accept": float((in_scores >= threshold).mean()),
    }


def evaluate_methods(val_logits, val_y, oos_val_logits, test_logits, oos_test_logits, keep=0.95, method="auto"):
    """Fit T, compare all methods, pick one (auto = best AUROC on OOS validation)."""
    T = fit_temperature(val_logits, val_y)
    results = {}
    for m in METHODS:
        t = 1.0 if m == "msp" else T
        sv, so_v = score(val_logits, m, t), score(oos_val_logits, m, t)
        thr = threshold_for(sv, keep)
        results[m] = {
            "temperature": t, "threshold": thr,
            "val": ood_metrics(sv, so_v, thr),
            "test": ood_metrics(score(test_logits, m, t), score(oos_test_logits, m, t), thr),
        }
    chosen = max(METHODS, key=lambda m: results[m]["val"]["auroc"]) if method == "auto" else method
    return {"temperature": T, "chosen": chosen, "methods": results}


def expected_calibration_error(probs, y, bins=15):
    """ECE: average gap between confidence and accuracy (lower = better calibrated)."""
    conf, pred = probs.max(1), probs.argmax(1)
    correct = pred == np.asarray(y)
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)
