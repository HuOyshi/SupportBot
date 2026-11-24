"""
SupportBot web app (Streamlit).

    streamlit run app/app.py            ->  http://localhost:8501

Type one customer message, or upload a CSV / TXT file of messages, and see the detected intent,
confidence, out-of-scope flag, escalation recommendation and a suggested reply.
"""

import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from predict import SupportBot  # noqa: E402

MODEL_DIR = Path(os.environ.get("SUPPORTBOT_MODEL_DIR", ROOT / "models" / "best"))
EXAMPLES = ["I want to cancel my order", "I was charged twice!!! this is unacceptable",
            "hw do i trak my ordr", "can i pay with paypal?", "what's the weather tomorrow?"]

st.set_page_config(page_title="SupportBot", page_icon="💬", layout="wide")


@st.cache_resource(show_spinner="Loading model ...")
def get_bot():
    return SupportBot(MODEL_DIR)


def to_table(results):
    df = pd.DataFrame(results)
    df["top_intents"] = df["top_intents"].apply(lambda t: ", ".join(f"{i} ({p:.2f})" for i, p in t))
    return df[["text", "intent", "category", "confidence", "out_of_scope", "escalate",
               "escalation_reason", "top_intents", "response"]]


try:
    bot = get_bot()
except FileNotFoundError as e:
    st.error(f"{e}")
    st.stop()

info, m = bot.info, bot.info["metrics"]

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Model")
    st.write(f"**{info['run_name']}**")
    st.caption(f"MLflow run `{info['run_id'][:12]}`")
    c1, c2 = st.columns(2)
    c1.metric("Test accuracy", f"{m['test_accuracy']:.3f}")
    c2.metric("Test macro-F1", f"{m['test_macro_f1']:.3f}")
    c1.metric("Typo test F1", f"{m['test_noisy_macro_f1']:.3f}")
    c2.metric("OOS AUROC", f"{m['oos_test_auroc']:.3f}")
    st.caption(f"Out-of-scope method: **{info['ood_method']}**, threshold {info['ood_threshold']:.3f}. "
               f"Confidence is calibrated with temperature T = {info['temperature']:.2f}.")
    st.header("Escalation rules")
    st.markdown(
        f"- message is out of scope\n"
        f"- confidence below **{bot.cfg['low_confidence']}**\n"
        f"- sensitive intent: {', '.join(bot.cfg['escalate_intents'])}\n"
        f"- frustration words (e.g. *{', '.join(bot.cfg['frustration_words'][:4])}*)")

st.title("💬 SupportBot")
st.caption("Intent-aware customer support assistant · 27 intents · out-of-scope detection · escalation to a human agent")

tab_chat, tab_batch = st.tabs(["Single message", "Upload a file"])

# ---------------------------------------------------------------- single message
with tab_chat:
    st.write("Try an example:")
    cols = st.columns(len(EXAMPLES))
    for col, ex in zip(cols, EXAMPLES):
        if col.button(ex, width="stretch"):
            st.session_state["msg"] = ex
    text = st.text_area("Customer message", key="msg", height=90,
                        placeholder="e.g. I need to change the shipping address of my order")
    if st.button("Analyze", type="primary") and text.strip():
        r = bot.predict([text])[0]
        a, b, c = st.columns(3)
        a.metric("Detected intent", r["intent"])
        b.metric("Category", r["category"])
        c.metric("Confidence", f"{r['confidence']:.1%}")

        if r["out_of_scope"]:
            st.warning("**Out of scope:** this does not look like a customer-support request.")
        if r["escalate"]:
            st.error(f"**Escalate to a human agent** - {r['escalation_reason']}")
        else:
            st.success("**No escalation needed** - the bot can answer this.")

        left, right = st.columns([3, 2])
        with left:
            st.subheader("Suggested reply")
            with st.chat_message("assistant"):
                st.write(r["response"])
            st.caption("Text in [brackets] is a placeholder for order-specific details.")
        with right:
            st.subheader("Top intents")
            for intent, p in r["top_intents"]:
                st.write(f"{intent} — {p:.1%}")
                st.progress(min(max(p, 0.0), 1.0))

# ---------------------------------------------------------------- batch upload
with tab_batch:
    st.write("Upload a **CSV** (column `text`, or the first column is used) or a **TXT** file (one message per line).")
    up = st.file_uploader("Customer messages", type=["csv", "txt"])
    if up is not None:
        if up.name.lower().endswith(".csv"):
            df_in = pd.read_csv(up)
            col = "text" if "text" in df_in.columns else df_in.columns[0]
            texts = df_in[col].dropna().astype(str).tolist()
        else:
            texts = [l.strip() for l in up.read().decode("utf-8", errors="ignore").splitlines() if l.strip()]
        if len(texts) > 5000:
            st.info("Only the first 5,000 messages are processed.")
            texts = texts[:5000]
        with st.spinner(f"Analyzing {len(texts)} messages ..."):
            res = to_table(bot.predict(texts))
        a, b, c = st.columns(3)
        a.metric("Messages", len(res))
        b.metric("Escalated", f"{res['escalate'].mean():.0%}")
        c.metric("Out of scope", f"{res['out_of_scope'].mean():.0%}")
        st.subheader("Intent distribution")
        st.bar_chart(res["intent"].value_counts(), horizontal=True)
        st.subheader("Results")
        st.dataframe(res, width="stretch", hide_index=True)
        st.download_button("Download results (CSV)", res.to_csv(index=False).encode("utf-8"),
                           "supportbot_predictions.csv", "text/csv")
