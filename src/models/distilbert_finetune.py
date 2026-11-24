"""
Optional: fine-tune a pre-trained DistilBERT (transfer learning) for comparison.
Disabled by default in configs/config.yaml because it is slow on a laptop CPU.
Needs:  pip install transformers
"""

import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score

from common import save_json, load_json


class DistilBertFinetune:
    def __init__(self, params, labels):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.params, self.labels = dict(params), list(labels)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(params["model"])
        self.net = AutoModelForSequenceClassification.from_pretrained(
            params["model"], num_labels=len(labels)).to(self.device)

    def _enc(self, texts):
        return self.tok(list(texts), padding=True, truncation=True,
                        max_length=self.params["max_len"], return_tensors="pt").to(self.device)

    def fit(self, X, y, X_val, y_val, log_fn=None):
        p = self.params
        pairs = list(zip(X, y))
        if p.get("max_train_samples") and len(pairs) > p["max_train_samples"]:
            pairs = random.Random(0).sample(pairs, p["max_train_samples"])
        opt = torch.optim.AdamW(self.net.parameters(), lr=p["lr"])
        steps = p["epochs"] * math.ceil(len(pairs) / p["batch_size"])
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: max(0.0, 1 - s / steps))
        history = []
        for epoch in range(1, p["epochs"] + 1):
            random.shuffle(pairs)
            self.net.train()
            total = 0.0
            for s in range(0, len(pairs), p["batch_size"]):
                bx, by = zip(*pairs[s:s + p["batch_size"]])
                out = self.net(**self._enc(bx), labels=torch.tensor(by, device=self.device))
                opt.zero_grad(); out.loss.backward(); opt.step(); sched.step()
                total += out.loss.item() * len(by)
            val_logits = self.logits(X_val)
            row = {"epoch": epoch, "train_loss": total / len(pairs),
                   "val_loss": F.cross_entropy(torch.from_numpy(val_logits), torch.tensor(np.array(y_val))).item(),
                   "val_accuracy": float((val_logits.argmax(1) == np.asarray(y_val)).mean()),
                   "val_macro_f1": float(f1_score(y_val, val_logits.argmax(1), average="macro"))}
            history.append(row)
            print(f"  epoch {epoch} | train loss {row['train_loss']:.4f} | val F1 {row['val_macro_f1']:.4f}")
            if log_fn:
                log_fn(row, epoch)
        return {"history": history, "epochs_trained": len(history)}

    @torch.no_grad()
    def logits(self, texts):
        self.net.eval()
        texts = list(texts)
        outs = [self.net(**self._enc(texts[s:s + 128])).logits.float().cpu() for s in range(0, len(texts), 128)]
        return torch.cat(outs).numpy()

    def save(self, d):
        d = Path(d); d.mkdir(parents=True, exist_ok=True)
        self.net.save_pretrained(d / "hf"); self.tok.save_pretrained(d / "hf")
        save_json({"params": self.params, "labels": self.labels}, d / "weights_meta.json")

    @classmethod
    def load(cls, d):
        meta = load_json(Path(d) / "weights_meta.json")
        params = dict(meta["params"], model=str(Path(d) / "hf"))
        return cls(params, meta["labels"])
