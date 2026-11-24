# Evaluation of the best model: `tfidf_logreg`

MLflow run id: `b8d07bc776094d5fae39132bc213c9ea`  |  OOS method: `energy`  |  temperature T = 0.516

| Metric | Value |
|---|---|
| Test accuracy | 0.9976 |
| Test macro-F1 | 0.9976 |
| Test weighted-F1 | 0.9976 |
| Noisy test accuracy (typos) | 0.9923 |
| Noisy test macro-F1 (typos) | 0.9921 |
| ECE before temperature scaling | 0.0089 |
| ECE after temperature scaling | 0.0014 |
| Out-of-scope AUROC | 0.9973 |
| Out-of-scope FPR@95%TPR | 0.0060 |
| Out-of-scope recall at threshold | 0.9940 |
| In-scope acceptance rate | 0.9566 |
| End-to-end in-scope accuracy | 0.9550 |

## All runs (from train.py)

| run | val_macro_f1 | test_accuracy | test_macro_f1 | test_noisy_macro_f1 | oos_test_auroc | oos_test_fpr_at_95tpr | train_time_sec | latency_ms_per_query |
|---|---|---|---|---|---|---|---|---|
| tfidf_logreg | 0.9980 | 0.9976 | 0.9976 | 0.9921 | 0.9973 | 0.0060 | 8.9162 | 1.2500 |
| naive_bayes_scratch | 0.9859 | 0.9903 | 0.9897 | 0.9650 | 0.9803 | 0.0870 | 0.8751 | 0.3125 |
| transformer_scratch_word | 0.9921 | 0.9955 | 0.9950 | 0.9527 | 0.9843 | 0.0440 | 457.3131 | 2.9687 |
| transformer_scratch_subword | 0.9975 | 0.9984 | 0.9983 | 0.9898 | 0.9983 | 0.0030 | 362.1953 | 2.4999 |

## Out-of-scope AUROC by detection method (test)

| run | msp | temp_msp | energy |
|---|---|---|---|
| tfidf_logreg | 0.9976 | 0.9966 | 0.9973 |
| naive_bayes_scratch | 0.9777 | 0.9803 | 0.1051 |
| transformer_scratch_word | 0.9831 | 0.9843 | 0.9758 |
| transformer_scratch_subword | 0.9982 | 0.9979 | 0.9983 |
