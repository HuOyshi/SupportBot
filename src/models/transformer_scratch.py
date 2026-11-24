"""
Transformer encoder intent classifier written from scratch in PyTorch.

Only basic building blocks are taken from PyTorch (nn.Linear, nn.Embedding, nn.LayerNorm).
Multi-head self-attention, the encoder block, pooling and the training loop are written here.

    tokens -> word embedding (+ optional sub-word embedding) + position embedding
           -> N x [ LayerNorm -> multi-head self-attention -> residual
                    LayerNorm -> feed-forward network     -> residual ]
           -> LayerNorm -> masked mean pooling -> linear layer -> one score per intent

Sub-word option (use_char_ngrams=True, fastText-style): every word is also represented by the
sum of its character 3-5-gram embeddings (hashed into a fixed number of buckets). A misspelled
word like "refnd" still shares most n-grams with "refund", which makes the model more robust
to typos and to words it never saw in training.
"""

import copy
import math
import time
import zlib
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score

from common import tokenize, save_json, load_json

PAD, UNK = 0, 1
MAX_NGRAMS_PER_WORD = 16


def char_ngram_ids(word, buckets):
    """Hashed character 3..5-grams of '<word>' -> ids in [1, buckets] (0 = padding)."""
    w = f"<{word}>"
    grams = [w[i:i + n] for n in (3, 4, 5) for i in range(len(w) - n + 1)]
    return [zlib.crc32(g.encode("utf-8")) % buckets + 1 for g in grams[:MAX_NGRAMS_PER_WORD]]


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.h, self.dk = n_heads, d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)   # queries, keys and values in one projection
        self.out = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, pad_mask):
        # x: [batch, seq, d_model]   pad_mask: [batch, seq] True where the token is padding
        b, s, d = x.shape
        q, k, v = self.qkv(x).view(b, s, 3, self.h, self.dk).permute(2, 0, 3, 1, 4)  # each [b, h, s, dk]
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.dk)                         # [b, h, s, s]
        scores = scores.masked_fill(pad_mask[:, None, None, :], float("-inf"))        # ignore padding
        attn = self.drop(torch.softmax(scores, dim=-1))
        ctx = (attn @ v).transpose(1, 2).reshape(b, s, d)                             # join the heads
        return self.out(ctx)


class EncoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, ff_dim, dropout):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.ff = nn.Sequential(nn.Linear(d_model, ff_dim), nn.GELU(), nn.Dropout(dropout),
                                nn.Linear(ff_dim, d_model))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, pad_mask):
        x = x + self.drop(self.attn(self.ln1(x), pad_mask))   # pre-norm residual connections
        x = x + self.drop(self.ff(self.ln2(x)))
        return x


class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, n_classes, p):
        super().__init__()
        d = p["d_model"]
        self.use_char = p["use_char_ngrams"]
        self.word_emb = nn.Embedding(vocab_size, d, padding_idx=PAD)
        if self.use_char:
            self.char_emb = nn.Embedding(p["char_buckets"] + 1, d, padding_idx=0)
        self.pos_emb = nn.Embedding(p["max_len"], d)
        self.drop = nn.Dropout(p["dropout"])
        self.blocks = nn.ModuleList([EncoderBlock(d, p["n_heads"], p["ff_dim"], p["dropout"])
                                     for _ in range(p["n_layers"])])
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, n_classes)

    def forward(self, ids, char_ids=None):
        pad_mask = ids == PAD
        pos = torch.arange(ids.size(1), device=ids.device)
        x = self.word_emb(ids)
        if self.use_char:
            # sum of the word's character n-gram vectors (padding vector is zero)
            x = x + self.char_emb(char_ids).sum(dim=2)
        x = self.drop(x + self.pos_emb(pos)[None])
        for block in self.blocks:
            x = block(x, pad_mask)
        x = self.ln(x)
        keep = (~pad_mask).unsqueeze(-1).float()
        pooled = (x * keep).sum(1) / keep.sum(1).clamp(min=1.0)   # average over real tokens only
        return self.head(pooled)


class TransformerScratch:
    def __init__(self, params, labels):
        self.params, self.labels = dict(params), list(labels)
        self.vocab = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = None

    # ---- text -> tensors ----------------------------------------------------
    def _build_vocab(self, texts):
        counts = Counter(tok for t in texts for tok in tokenize(t))
        min_freq = self.params.get("min_freq", 1)
        self.vocab = {"<pad>": PAD, "<unk>": UNK}
        for tok, c in counts.most_common():
            if c >= min_freq:
                self.vocab[tok] = len(self.vocab)

    def _encode(self, texts):
        L, buckets = self.params["max_len"], self.params["char_buckets"]
        ids = np.zeros((len(texts), L), dtype=np.int64)
        char_ids = np.zeros((len(texts), L, MAX_NGRAMS_PER_WORD), dtype=np.int64) if self.params["use_char_ngrams"] else None
        for i, t in enumerate(texts):
            for j, tok in enumerate(tokenize(t)[:L]):
                ids[i, j] = self.vocab.get(tok, UNK)
                if char_ids is not None:
                    g = char_ngram_ids(tok, buckets)
                    char_ids[i, j, :len(g)] = g
        ids = torch.from_numpy(ids)
        return ids, (torch.from_numpy(char_ids) if char_ids is not None else None)

    def _batches(self, ids, char_ids, y=None, shuffle=False, bs=None):
        bs = bs or self.params["batch_size"]
        order = torch.randperm(len(ids)) if shuffle else torch.arange(len(ids))
        for s in range(0, len(ids), bs):
            idx = order[s:s + bs]
            out = [ids[idx].to(self.device), char_ids[idx].to(self.device) if char_ids is not None else None]
            if y is not None:
                out.append(y[idx].to(self.device))
            yield out

    # ---- training -------------------------------------------------------------
    def fit(self, X, y, X_val, y_val, log_fn=None):
        p = self.params
        self._build_vocab(X)
        self.net = TransformerClassifier(len(self.vocab), len(self.labels), p).to(self.device)
        ids, cids = self._encode(X)
        y_t = torch.tensor(np.array(y), dtype=torch.long)
        opt = torch.optim.AdamW(self.net.parameters(), lr=p["lr"], weight_decay=p["weight_decay"])
        steps = p["epochs"] * math.ceil(len(ids) / p["batch_size"])
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=p["lr"], total_steps=steps, pct_start=0.1)

        history, best_f1, best_state, bad_epochs = [], -1.0, None, 0
        for epoch in range(1, p["epochs"] + 1):
            t0 = time.time()
            self.net.train()
            total, n = 0.0, 0
            for bid, bcid, by in self._batches(ids, cids, y_t, shuffle=True):
                loss = F.cross_entropy(self.net(bid, bcid), by, label_smoothing=p["label_smoothing"])
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                opt.step(); sched.step()
                total += loss.item() * len(by); n += len(by)

            val_logits = self.logits(X_val)
            val_pred = val_logits.argmax(1)
            val_loss = F.cross_entropy(torch.from_numpy(val_logits), torch.tensor(np.array(y_val))).item()
            row = {"epoch": epoch, "train_loss": total / n, "val_loss": val_loss,
                   "val_accuracy": float((val_pred == np.asarray(y_val)).mean()),
                   "val_macro_f1": float(f1_score(y_val, val_pred, average="macro")),
                   "epoch_seconds": time.time() - t0}
            history.append(row)
            print(f"  epoch {epoch:2d} | train loss {row['train_loss']:.4f} | val loss {val_loss:.4f} "
                  f"| val acc {row['val_accuracy']:.4f} | val F1 {row['val_macro_f1']:.4f}")
            if log_fn:
                log_fn(row, epoch)

            # early stopping: keep the weights of the best validation epoch
            if row["val_macro_f1"] > best_f1 + 1e-4:
                best_f1, best_state, bad_epochs = row["val_macro_f1"], copy.deepcopy(self.net.state_dict()), 0
            else:
                bad_epochs += 1
                if bad_epochs >= p["patience"]:
                    print(f"  early stopping after epoch {epoch}")
                    break
        self.net.load_state_dict(best_state)
        return {"history": history, "best_val_macro_f1": best_f1, "epochs_trained": len(history)}

    @torch.no_grad()
    def logits(self, texts):
        self.net.eval()
        ids, cids = self._encode(list(texts))
        outs = [self.net(bid, bcid).float().cpu() for bid, bcid in self._batches(ids, cids, bs=512)]
        return torch.cat(outs).numpy() if outs else np.zeros((0, len(self.labels)), dtype=np.float32)

    def n_parameters(self):
        return sum(p.numel() for p in self.net.parameters())

    def save(self, d):
        d = Path(d); d.mkdir(parents=True, exist_ok=True)
        torch.save({k: v.float().cpu() for k, v in self.net.state_dict().items()}, d / "transformer.pt")
        save_json({"params": self.params, "labels": self.labels, "vocab": self.vocab}, d / "weights_meta.json")

    @classmethod
    def load(cls, d):
        meta = load_json(Path(d) / "weights_meta.json")
        m = cls(meta["params"], meta["labels"])
        m.vocab = meta["vocab"]
        m.net = TransformerClassifier(len(m.vocab), len(m.labels), m.params).to(m.device)
        m.net.load_state_dict(torch.load(Path(d) / "transformer.pt", map_location=m.device))
        m.net.eval()
        return m
