"""
Full SupportBot prediction pipeline (used by the CLI below and by the Streamlit app).

    python src/predict.py --input "I want to cancel my order"
    python src/predict.py --input samples/sample_queries.csv --output results/predictions.csv

For every customer message it returns:
  intent + category, calibrated confidence, top-3 intents, out-of-scope flag,
  escalation recommendation (+ reason) and a suggested reply.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import BEST_DIR, load_config, normalize, tokenize  # noqa: E402
from models import load_model  # noqa: E402
from ood import calibrated_probs, score  # noqa: E402
from responder import OOS_REPLY, Responder  # noqa: E402


class SupportBot:
    def __init__(self, model_dir=BEST_DIR, config=None):
        model_dir = Path(model_dir)
        if not (model_dir / "model_info.json").exists():
            raise FileNotFoundError(f"No trained model in {model_dir}. Run: python src/train.py")
        self.cfg = load_config(config)["predict"]
        self.model, self.info = load_model(model_dir)
        self.labels = self.info["labels"]
        self.responder = Responder.load(model_dir)

    def predict(self, texts):
        texts = [str(t) for t in texts]
        if not texts:
            return []
        logits = self.model.logits(texts)
        probs = calibrated_probs(logits, self.info["temperature"])        # calibrated confidence
        ood_s = score(logits, self.info["ood_method"], self.info["ood_temperature"])
        k = self.cfg.get("top_k", 3)
        out = []
        for i, text in enumerate(texts):
            order = np.argsort(-probs[i])[:k]
            intent, conf = self.labels[order[0]], float(probs[i, order[0]])
            oos = bool(ood_s[i] < self.info["ood_threshold"]) or not text.strip()

            reasons = []
            if oos:
                reasons.append("out of scope")
            elif conf < self.cfg["low_confidence"]:
                reasons.append(f"low confidence ({conf:.2f})")
            if not oos and intent in self.cfg["escalate_intents"]:
                reasons.append(f"sensitive intent ({intent})")
            low, toks = normalize(text), set(tokenize(text))
            hits = [w for w in self.cfg["frustration_words"] if (w in toks if w.isalpha() else w in low)]
            if hits:
                reasons.append("frustrated customer (" + ", ".join(hits) + ")")

            out.append({
                "text": text,
                "intent": "out_of_scope" if oos else intent,
                "category": "-" if oos else self.info["category"].get(intent, ""),
                "confidence": round(conf, 4),
                "top_intents": [(self.labels[j], round(float(probs[i, j]), 4)) for j in order],
                "out_of_scope": oos,
                "ood_score": round(float(ood_s[i]), 4),
                "escalate": bool(reasons),
                "escalation_reason": "; ".join(reasons) if reasons else "",
                "response": OOS_REPLY if oos else self.responder.reply(text, intent),
            })
        return out


def read_inputs(arg):
    """Text, a .txt file (one message per line) or a .csv file (column 'text' or the first column)."""
    p = Path(arg)
    if p.suffix.lower() == ".csv" and p.exists():
        df = pd.read_csv(p)
        col = "text" if "text" in df.columns else df.columns[0]
        return df[col].astype(str).tolist()
    if p.suffix.lower() == ".txt" and p.exists():
        return [l.strip() for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    return [arg]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="a message, or a .csv / .txt file of messages")
    ap.add_argument("--model-dir", default=str(BEST_DIR))
    ap.add_argument("--output", help="optional CSV to save the predictions")
    args = ap.parse_args()

    bot = SupportBot(args.model_dir)
    results = bot.predict(read_inputs(args.input))
    for r in results:
        print("-" * 80)
        print(f"Message   : {r['text']}")
        print(f"Intent    : {r['intent']} ({r['category']})   confidence {r['confidence']:.2f}")
        print(f"Top-3     : " + ", ".join(f"{i} {p:.2f}" for i, p in r["top_intents"]))
        print(f"Escalate  : {'YES - ' + r['escalation_reason'] if r['escalate'] else 'no'}")
        print(f"Reply     : {r['response'][:300]}{'...' if len(r['response']) > 300 else ''}")
    if args.output:
        df = pd.DataFrame(results)
        df["top_intents"] = df["top_intents"].apply(lambda t: "; ".join(f"{i}:{p}" for i, p in t))
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.output, index=False)
        print(f"\nSaved {len(df)} predictions to {args.output}")


if __name__ == "__main__":
    main()
