"""
Sentiment Analysis Web App — comparing 4 models in a 2x2 grid:

                | Original CNN         | Improved CNN
    ------------+----------------------+---------------------
    GloVe 100d  | glove_original       | glove_improved
    Word2Vec 300| word2vec_original    | word2vec_improved

The app loads only the 4 saved .keras files + tokenizer + config + results.
The 130 MB GloVe file and the 3.5 GB Word2Vec binary are NOT needed at
runtime — their values are already baked into the trained model weights.
"""
import os
import json
import pickle
import re
import numpy as np
import pandas as pd
import streamlit as st
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

st.set_page_config(page_title="GloVe vs Word2Vec — Sentiment CNN", page_icon="🎬", layout="wide")

ART_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")

STOPWORDS = set("""
a an the and or but if while of at by for with about against between into through during
before after above below to from up down in out on off over under again further then once
here there when where why how all any both each few more most other some such no nor not
only own same so than too very s t can will just don should now i me my myself we our ours
ourselves you your yours yourself yourselves he him his himself she her hers herself it its
itself they them their theirs themselves what which who whom this that these those am is
are was were be been being have has had having do does did doing would could should may
might must shall ll re ve d m
""".split())

def clean_text(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = [t for t in text.split() if t and t not in STOPWORDS and len(t) > 1]
    return " ".join(tokens)

# Friendly labels (kept consistent everywhere in the UI)
LABELS = {
    "glove_original":    "GloVe + Original CNN",
    "glove_improved":    "GloVe + Improved CNN",
    "word2vec_original": "Word2Vec + Original CNN",
    "word2vec_improved": "Word2Vec + Improved CNN",
}
ORDER = ["glove_original", "glove_improved", "word2vec_original", "word2vec_improved"]

@st.cache_resource
def load_all():
    with open(os.path.join(ART_DIR, "tokenizer.pkl"), "rb") as f:
        tokenizer = pickle.load(f)
    with open(os.path.join(ART_DIR, "config.json")) as f:
        config = json.load(f)
    with open(os.path.join(ART_DIR, "results.json")) as f:
        results = json.load(f)
    models = {}
    for key, meta in config["models"].items():
        models[key] = load_model(os.path.join(ART_DIR, meta["file"]))
    return tokenizer, config, results, models

def predict_sentiment(texts, model, tokenizer, maxlen):
    cleaned = [clean_text(t) for t in texts]
    seqs = tokenizer.texts_to_sequences(cleaned)
    X = pad_sequences(seqs, maxlen=maxlen)
    probs = model.predict(X, verbose=0)
    pos_p, neg_p = probs[:, 0], probs[:, 1]
    total = pos_p + neg_p + 1e-9
    pos_norm = pos_p / total
    labels = np.where(pos_norm >= 0.5, "Positive", "Negative")
    confidences = np.where(pos_norm >= 0.5, pos_norm, 1 - pos_norm)
    return labels, confidences, pos_norm

# -----------------------------
# Header
# -----------------------------
st.title("🎬 Sentiment CNN — GloVe vs Word2Vec")
st.caption("Trained on IMDB · Cross-domain tested on TripAdvisor · 4 models compared")

try:
    tokenizer, config, results, models = load_all()
except FileNotFoundError as e:
    st.error(f"Model artifacts not found ({e}). Run `python train_models.py` first.")
    st.stop()

# -----------------------------
# Sidebar — choose model
# -----------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    st.markdown("**Choose embedding**")
    emb_choice = st.radio(" ", ["GloVe 100d", "Word2Vec 300d"], index=0, label_visibility="collapsed")
    st.markdown("**Choose architecture**")
    arch_choice = st.radio("  ", ["Improved CNN (recommended)", "Original CNN (from lab)"],
                           index=0, label_visibility="collapsed")

    emb_key  = "glove" if emb_choice.startswith("GloVe") else "word2vec"
    arch_key = "improved" if arch_choice.startswith("Improved") else "original"
    active_key   = f"{emb_key}_{arch_key}"
    active_model = models[active_key]
    active_meta  = config["models"][active_key]
    active_maxlen = active_meta["maxlen"]

    st.markdown("---")
    st.subheader("📊 This model's metrics")
    r = results[active_key]
    st.metric("IMDB accuracy",         f"{r['imdb_test']['accuracy']*100:.2f}%")
    st.metric("IMDB loss",             f"{r['imdb_test']['loss']:.4f}")
    st.metric("TripAdvisor accuracy",  f"{r['tripadvisor']['accuracy']*100:.2f}%")
    st.metric("TripAdvisor loss",      f"{r['tripadvisor']['loss']:.4f}")

    st.markdown("---")
    st.caption(f"**{LABELS[active_key]}**")
    st.caption(f"Sequence length: {active_maxlen}")
    st.caption(f"Embedding: {active_meta['embedding']}")
    st.caption(f"Architecture: {active_meta['arch']}")
    st.caption(f"Vocab: {config['vocab_size']}")
    if emb_key == "glove":
        st.caption(f"GloVe coverage: {config.get('glove_coverage_pct','?')}%")
    else:
        st.caption(f"Word2Vec coverage: {config.get('word2vec_coverage_pct','?')}%")

# -----------------------------
# Tabs
# -----------------------------
tab_single, tab_batch, tab_compare, tab_all = st.tabs(
    ["🔤 Single sentence", "📂 CSV upload", "📈 4-way comparison", "🎯 Predict with all 4"]
)

# ---------- Single sentence ----------
with tab_single:
    st.subheader("Predict sentiment for a single sentence")
    default_text = ("The hotel was absolutely wonderful — clean rooms, friendly staff, "
                    "and a great location. I'll definitely come back!")
    text_input = st.text_area("Enter a sentence or short review:", value=default_text, height=120)
    if st.button("Predict", type="primary", key="single_predict"):
        if not text_input.strip():
            st.warning("Please enter some text.")
        else:
            labels, confs, pos_norm = predict_sentiment([text_input], active_model, tokenizer, active_maxlen)
            label, conf, pos_p = labels[0], confs[0], pos_norm[0]
            color = "🟢" if label == "Positive" else "🔴"
            st.markdown(f"### {color} Prediction: **{label}**  *(via {LABELS[active_key]})*")
            st.metric("Confidence", f"{conf*100:.1f}%")
            st.progress(float(pos_p),     text=f"Positive: {pos_p*100:.1f}%")
            st.progress(float(1 - pos_p), text=f"Negative: {(1-pos_p)*100:.1f}%")
            with st.expander("🔍 Preprocessing details"):
                cleaned = clean_text(text_input)
                st.write("**Cleaned text:**")
                st.code(cleaned or "(empty after cleaning)")
                seq = tokenizer.texts_to_sequences([cleaned])
                st.write(f"**Tokens:** {len(seq[0])} (padded/truncated to {active_maxlen})")

# ---------- Batch CSV ----------
with tab_batch:
    st.subheader("Batch prediction — upload a CSV")
    st.caption("Predictions use the model selected in the sidebar.")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded:
        try:
            df = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"Couldn't read CSV: {e}")
            st.stop()
        st.write(f"Loaded **{len(df)}** rows. Preview:")
        st.dataframe(df.head(), use_container_width=True)
        text_col = st.selectbox("Text column:", options=list(df.columns), index=0)
        possible = [c for c in df.columns if c != text_col]
        label_col = st.selectbox("Optional ground-truth column:", options=["(none)"] + possible, index=0)
        max_rows = st.number_input("Max rows to predict", min_value=1, max_value=len(df),
                                   value=min(500, len(df)))
        if st.button("Run predictions", type="primary", key="batch_predict"):
            sub = df.head(int(max_rows)).copy()
            with st.spinner(f"Predicting {len(sub)} rows with {LABELS[active_key]}..."):
                labels, confs, pos_norm = predict_sentiment(
                    sub[text_col].astype(str).tolist(), active_model, tokenizer, active_maxlen)
            sub["predicted_sentiment"] = labels
            sub["confidence"]          = np.round(confs, 4)
            sub["pos_probability"]     = np.round(pos_norm, 4)
            st.success(f"Done.")
            counts = pd.Series(labels).value_counts()
            c1, c2 = st.columns(2)
            c1.metric("Positive", int(counts.get("Positive", 0)))
            c2.metric("Negative", int(counts.get("Negative", 0)))
            if label_col != "(none)":
                def to_binary(v):
                    if isinstance(v, str):
                        s = v.strip().lower()
                        if s in {"1", "pos", "positive", "true"}:  return 1
                        if s in {"0", "neg", "negative", "false"}: return 0
                        try:
                            iv = int(s)
                            return 1 if iv >= 4 else (0 if iv <= 2 else None)
                        except ValueError: return None
                    try: iv = int(v)
                    except (TypeError, ValueError): return None
                    if iv in (0, 1): return iv
                    return 1 if iv >= 4 else (0 if iv <= 2 else None)
                gt = sub[label_col].apply(to_binary)
                pred_bin = (sub["predicted_sentiment"] == "Positive").astype(int)
                mask = gt.notna()
                if mask.sum() > 0:
                    acc = (gt[mask] == pred_bin[mask]).mean()
                    st.metric(f"Accuracy vs '{label_col}'", f"{acc*100:.2f}%",
                              help=f"On {int(mask.sum())} rows where labels parsed.")
            st.dataframe(sub, use_container_width=True)
            csv_bytes = sub.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download predictions CSV", data=csv_bytes,
                               file_name="predictions.csv", mime="text/csv")

# ---------- 4-way comparison ----------
with tab_compare:
    st.subheader("📈 All four models — side by side")
    st.write("Each model was trained on **IMDB labelled sentences** with its embedding "
             "*frozen* (so what we're really comparing are the embedding spaces and "
             "the CNN architectures). All four are then tested on **TripAdvisor hotel reviews**, "
             "a totally different domain.")

    rows = []
    for key in ORDER:
        r = results[key]
        rows.append({
            "Model": LABELS[key],
            "Embedding":  config["models"][key]["embedding"],
            "Architecture": config["models"][key]["arch"],
            "IMDB acc":         f"{r['imdb_test']['accuracy']*100:.2f}%",
            "IMDB loss":        f"{r['imdb_test']['loss']:.4f}",
            "TripAdvisor acc":  f"{r['tripadvisor']['accuracy']*100:.2f}%",
            "TripAdvisor loss": f"{r['tripadvisor']['loss']:.4f}",
        })
    cmp_df = pd.DataFrame(rows)
    st.dataframe(cmp_df, use_container_width=True, hide_index=True)

    # Bar charts using just Streamlit's built-in (no extra deps)
    st.markdown("#### Accuracy by model")
    chart_df = pd.DataFrame({
        "IMDB":        [results[k]["imdb_test"]["accuracy"]*100   for k in ORDER],
        "TripAdvisor": [results[k]["tripadvisor"]["accuracy"]*100 for k in ORDER],
    }, index=[LABELS[k] for k in ORDER])
    st.bar_chart(chart_df)

    st.markdown("---")
    st.markdown("### What the 2×2 design lets us read off")
    st.markdown("""
**Rows = embedding choice (GloVe vs Word2Vec)** · **Columns = architecture (Original vs Improved)**

- Comparing **down a column** isolates the **embedding** effect (same architecture, different pretrained vectors).
- Comparing **across a row** isolates the **architecture** effect (same embeddings, different CNN).
- The **diagonal vs off-diagonal** tells us whether the two effects are independent
  or whether one embedding/architecture combo is especially strong.

**General expectations:**
1. **Word2Vec 300d** tends to encode richer semantics than GloVe 100d on natural-domain text,
   so Word2Vec models often generalize a bit better on TripAdvisor (cross-domain).
2. The **Improved CNN** (multi-branch kernels 2/3/4/5 + GlobalMaxPooling + stronger dropout)
   tends to beat the original on the small IMDB set regardless of embedding.
3. With **frozen embeddings**, the cross-domain gap (IMDB → TripAdvisor) is smaller than
   with trainable embeddings, because the embeddings haven't been over-fit to IMDB vocab.
    """)

# ---------- Predict with all 4 ----------
with tab_all:
    st.subheader("🎯 Run a single sentence through all four models")
    text2 = st.text_area("Sentence:", value="The room was small but the staff were lovely.",
                         height=100, key="all4_text")
    if st.button("Predict with all four", type="primary", key="all4_predict"):
        if not text2.strip():
            st.warning("Please enter some text.")
        else:
            rows = []
            for key in ORDER:
                meta = config["models"][key]
                labels, confs, pos_norm = predict_sentiment(
                    [text2], models[key], tokenizer, meta["maxlen"])
                rows.append({
                    "Model": LABELS[key],
                    "Prediction": labels[0],
                    "Confidence": f"{confs[0]*100:.1f}%",
                    "P(Positive)": f"{pos_norm[0]*100:.1f}%",
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            preds = [r["Prediction"] for r in rows]
            if len(set(preds)) == 1:
                st.success(f"All four models agree: **{preds[0]}** ✅")
            else:
                st.info("Models disagree — check the table above to see which combination "
                        "swings which way.")
