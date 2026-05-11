"""
Sentiment Analysis Web App (CNN)
- Predicts Positive / Negative for a single sentence
- Lets the user pick architecture (Original / Improved) and embedding
  (Scratch / GloVe / Word2Vec) — only modes whose models exist are shown
- CSV upload for batch prediction
- Comparison tab: architecture comparison AND embedding comparison
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

st.set_page_config(
    page_title="Sentiment Analysis (CNN)",
    page_icon="🎬",
    layout="wide",
)

ART_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")

# -----------------------------
# Preprocessing — must match train_models.py exactly
# -----------------------------
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
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = [t for t in text.split() if t and t not in STOPWORDS and len(t) > 1]
    return " ".join(tokens)

# -----------------------------
# Cached loaders
# -----------------------------
@st.cache_resource
def load_artifacts():
    with open(os.path.join(ART_DIR, "tokenizer.pkl"), "rb") as f:
        tokenizer = pickle.load(f)
    with open(os.path.join(ART_DIR, "config.json"), "r") as f:
        config = json.load(f)
    with open(os.path.join(ART_DIR, "results.json"), "r") as f:
        results = json.load(f)
    # Discover which model files actually exist
    available = {}
    for mode in config["available_modes"]:
        for arch in ("original", "improved"):
            path = os.path.join(ART_DIR, f"{arch}_cnn_{mode}.keras")
            if os.path.exists(path):
                available.setdefault(mode, {})[arch] = load_model(path)
    return tokenizer, config, results, available

def predict_sentiment(texts, model, tokenizer, maxlen):
    cleaned = [clean_text(t) for t in texts]
    seqs = tokenizer.texts_to_sequences(cleaned)
    X = pad_sequences(seqs, maxlen=maxlen)
    probs = model.predict(X, verbose=0)
    pos_probs = probs[:, 0]
    neg_probs = probs[:, 1]
    total = pos_probs + neg_probs + 1e-9
    pos_norm = pos_probs / total
    labels = np.where(pos_norm >= 0.5, "Positive", "Negative")
    confidences = np.where(pos_norm >= 0.5, pos_norm, 1 - pos_norm)
    return labels, confidences, pos_norm

MODE_LABELS = {
    "scratch":  "Trained from scratch",
    "glove":    "GloVe (pre-trained)",
    "word2vec": "Word2Vec (pre-trained)",
}
ARCH_LABELS = {
    "original": "Original CNN (lab)",
    "improved": "Improved CNN",
}

# =============================
# UI
# =============================
st.title("🎬 Sentiment Analysis Web App (CNN)")
st.caption("Trained on IMDB labelled sentences · Cross-domain tested on "
           "TripAdvisor hotel reviews · Compares Scratch / GloVe / Word2Vec embeddings")

try:
    tokenizer, config, results, available = load_artifacts()
except FileNotFoundError:
    st.error("❌ Model artifacts not found. Run `python3 train_models.py` first.")
    st.stop()

if not available:
    st.error("❌ No trained models found in `artifacts/`. "
             "Run `python3 train_models.py` first.")
    st.stop()

# Sidebar: pick architecture + embedding mode
with st.sidebar:
    st.header("⚙️ Model selection")

    arch_choice = st.radio(
        "Architecture",
        options=["improved", "original"],
        format_func=lambda k: ARCH_LABELS[k],
        index=0,
    )

    # Only offer embedding modes that actually have a trained model for this arch
    available_modes_for_arch = [m for m in config["available_modes"]
                                 if m in available and arch_choice in available[m]]
    if not available_modes_for_arch:
        st.error(f"No models found for architecture '{arch_choice}'.")
        st.stop()

    mode_choice = st.radio(
        "Embedding",
        options=available_modes_for_arch,
        format_func=lambda k: f"{MODE_LABELS[k]} (dim={config['embedding_dims'][k]})",
        index=0,
    )

    active_model = available[mode_choice][arch_choice]
    active_maxlen = (config["MAX_SEQUENCE_LENGTH_IMPROVED"]
                     if arch_choice == "improved"
                     else config["MAX_SEQUENCE_LENGTH_ORIG"])

    st.markdown("---")
    st.subheader("📊 Test metrics")
    m = results[mode_choice]["models"][arch_choice]
    st.metric("IMDB test accuracy",
              f"{m['imdb_test']['accuracy']*100:.2f}%")
    st.metric("IMDB test loss",
              f"{m['imdb_test']['loss']:.4f}")
    st.metric("TripAdvisor (cross-domain) accuracy",
              f"{m['tripadvisor']['accuracy']*100:.2f}%")
    st.metric("TripAdvisor (cross-domain) loss",
              f"{m['tripadvisor']['loss']:.4f}")

    st.markdown("---")
    st.caption(f"Embedding dim: {config['embedding_dims'][mode_choice]}")
    st.caption(f"Sequence length: {active_maxlen}")
    st.caption(f"Vocab size: {results[mode_choice]['vocab_size']}")

# Main tabs
tab_single, tab_batch, tab_arch, tab_embed = st.tabs([
    "🔤 Single sentence",
    "📂 CSV upload",
    "📈 Architecture comparison",
    "🧬 Embedding comparison",
])

# ---------- Tab 1: single sentence ----------
with tab_single:
    st.subheader("Predict sentiment for a single sentence")
    default_text = ("The hotel was absolutely wonderful — clean rooms, friendly "
                    "staff, and a great location. I'll definitely come back!")
    text_input = st.text_area("Enter a sentence or short review:",
                              value=default_text, height=120)
    if st.button("Predict", type="primary"):
        if not text_input.strip():
            st.warning("Please enter some text.")
        else:
            labels, confidences, pos_norm = predict_sentiment(
                [text_input], active_model, tokenizer, active_maxlen,
            )
            label = labels[0]
            confidence = confidences[0]
            pos_p = pos_norm[0]

            color = "🟢" if label == "Positive" else "🔴"
            st.markdown(f"### {color} Prediction: **{label}**")
            st.metric("Confidence", f"{confidence*100:.1f}%")

            st.write("Probability distribution:")
            st.progress(float(pos_p), text=f"Positive: {pos_p*100:.1f}%")
            st.progress(float(1 - pos_p),
                        text=f"Negative: {(1-pos_p)*100:.1f}%")

            with st.expander("🔍 Show preprocessing details"):
                cleaned = clean_text(text_input)
                st.write("**Cleaned text:**")
                st.code(cleaned or "(empty after cleaning)")
                seq = tokenizer.texts_to_sequences([cleaned])
                st.write(f"**Token sequence length:** {len(seq[0])} "
                         f"(padded/truncated to {active_maxlen})")

            # Show how OTHER available models would have predicted this
            with st.expander("🔬 How would the other models predict this?"):
                rows = []
                for mode in config["available_modes"]:
                    if mode not in available:
                        continue
                    for arch in ("original", "improved"):
                        if arch not in available[mode]:
                            continue
                        ml = (config["MAX_SEQUENCE_LENGTH_IMPROVED"]
                              if arch == "improved"
                              else config["MAX_SEQUENCE_LENGTH_ORIG"])
                        labels_o, confs_o, posn_o = predict_sentiment(
                            [text_input], available[mode][arch], tokenizer, ml,
                        )
                        rows.append({
                            "Architecture": ARCH_LABELS[arch],
                            "Embedding": MODE_LABELS[mode],
                            "Prediction": labels_o[0],
                            "Confidence": f"{confs_o[0]*100:.1f}%",
                            "P(positive)": f"{posn_o[0]*100:.1f}%",
                        })
                st.dataframe(pd.DataFrame(rows), use_container_width=True,
                             hide_index=True)

# ---------- Tab 2: CSV upload ----------
with tab_batch:
    st.subheader("Batch prediction — upload a CSV")
    st.caption("Your CSV must contain a text column. Predictions are added "
               "as new columns.")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded:
        try:
            df = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"Couldn't read CSV: {e}")
            st.stop()

        st.write(f"Loaded **{len(df)}** rows. Preview:")
        st.dataframe(df.head(), use_container_width=True)

        text_col = st.selectbox(
            "Which column contains the text?",
            options=list(df.columns), index=0,
        )
        possible_label_cols = [c for c in df.columns if c != text_col]
        label_col = st.selectbox(
            "Optional: ground-truth label column (skip if none)",
            options=["(none)"] + possible_label_cols, index=0,
        )
        max_rows = st.number_input(
            "Max rows to predict (cap for speed)",
            min_value=1, max_value=len(df), value=min(500, len(df)),
        )

        if st.button("Run predictions", type="primary"):
            sub = df.head(int(max_rows)).copy()
            with st.spinner(f"Predicting {len(sub)} rows..."):
                labels, confs, pos_norm = predict_sentiment(
                    sub[text_col].astype(str).tolist(),
                    active_model, tokenizer, active_maxlen,
                )
            sub["predicted_sentiment"] = labels
            sub["confidence"] = np.round(confs, 4)
            sub["pos_probability"] = np.round(pos_norm, 4)

            st.success(f"Done — predicted {len(sub)} rows.")
            counts = pd.Series(labels).value_counts()
            c1, c2 = st.columns(2)
            with c1:
                st.metric("Positive", int(counts.get("Positive", 0)))
            with c2:
                st.metric("Negative", int(counts.get("Negative", 0)))

            if label_col != "(none)":
                gt_raw = sub[label_col]
                def to_binary(v):
                    if isinstance(v, str):
                        s = v.strip().lower()
                        if s in {"1", "pos", "positive", "true"}: return 1
                        if s in {"0", "neg", "negative", "false"}: return 0
                        try:
                            iv = int(s)
                            return 1 if iv >= 4 else (0 if iv <= 2 else None)
                        except ValueError:
                            return None
                    try: iv = int(v)
                    except (TypeError, ValueError): return None
                    if iv in (0, 1): return iv
                    return 1 if iv >= 4 else (0 if iv <= 2 else None)
                gt = gt_raw.apply(to_binary)
                pred_bin = (sub["predicted_sentiment"] == "Positive").astype(int)
                mask = gt.notna()
                if mask.sum() > 0:
                    acc = (gt[mask] == pred_bin[mask]).mean()
                    st.metric(f"Accuracy vs '{label_col}'", f"{acc*100:.2f}%",
                              help=f"Computed on {int(mask.sum())} rows.")

            st.dataframe(sub, use_container_width=True)
            csv_bytes = sub.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download predictions CSV",
                data=csv_bytes, file_name="predictions.csv",
                mime="text/csv",
            )

# ---------- Tab 3: Architecture comparison ----------
with tab_arch:
    st.subheader("📈 Original CNN vs Improved CNN")
    st.write("Compares the two architectures using the **currently selected "
             f"embedding ({MODE_LABELS[mode_choice]})**.")

    if mode_choice in results:
        r = results[mode_choice]["models"]
        rows = []
        for arch in ("original", "improved"):
            if arch in r:
                rows.append({
                    "Architecture": ARCH_LABELS[arch],
                    "IMDB accuracy":   f"{r[arch]['imdb_test']['accuracy']*100:.2f}%",
                    "IMDB loss":       f"{r[arch]['imdb_test']['loss']:.4f}",
                    "TripAdvisor accuracy": f"{r[arch]['tripadvisor']['accuracy']*100:.2f}%",
                    "TripAdvisor loss":     f"{r[arch]['tripadvisor']['loss']:.4f}",
                })
        st.table(pd.DataFrame(rows))

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Original CNN (lab architecture)")
        st.markdown("""
- Conv1D(200, kernel=2) → MaxPool
- Conv1D(200, kernel=3) → MaxPool
- Flatten → Dropout(0.1) → Dense(128) → Dropout(0.1) → Dense(2, sigmoid)
- Sequence length: 50
        """)
    with c2:
        st.markdown("#### Improved CNN")
        st.markdown("""
- **Multi-branch** Conv1D with kernel sizes 2/3/4/5 (64 filters each)
- GlobalMaxPooling1D per branch → concatenate
- Dropout(0.4) → Dense(64) → Dropout(0.3) → Dense(2, sigmoid)
- Sequence length: 100
- **Stronger dropout** (0.4 / 0.3 vs 0.1 / 0.1) to reduce overfitting
        """)

    st.markdown("---")
    st.markdown("### Why the improved architecture wins")
    st.markdown("""
1. **Multi-branch kernels** capture 2/3/4/5-grams in parallel instead of
   sequentially, preserving information.
2. **GlobalMaxPooling1D** replaces `Flatten`, producing a far smaller dense
   input (~256 features vs ~2200) that overfits less.
3. **Stronger dropout** (0.4/0.3 vs 0.1/0.1) is a much better fit for the
   small (1000-sentence) IMDB dataset.
4. **Longer sequence length** (100 vs 50) captures more context for longer
   reviews in the cross-domain TripAdvisor test.
    """)

# ---------- Tab 4: Embedding comparison ----------
with tab_embed:
    st.subheader("🧬 Embedding strategy comparison")

    available_modes = [m for m in config["available_modes"] if m in available]

    if len(available_modes) < 2:
        st.info("ℹ️ Only one embedding mode is currently available "
                f"(**{MODE_LABELS[available_modes[0]]}**). To enable the "
                "GloVe vs Word2Vec comparison, download these files into "
                "the project folder and re-run `python3 train_models.py`:\n\n"
                "- `glove.6B.100d.txt` "
                "(from https://nlp.stanford.edu/projects/glove/)\n"
                "- `GoogleNews-vectors-negative300.bin.gz` "
                "(from https://code.google.com/archive/p/word2vec/)")
    else:
        st.write("Below: each row = embedding strategy, showing performance "
                 "for both architectures on IMDB and TripAdvisor.")

    # Build the comparison table for whatever is available
    rows = []
    for mode in available_modes:
        for arch in ("original", "improved"):
            if arch not in results.get(mode, {}).get("models", {}):
                continue
            m = results[mode]["models"][arch]
            rows.append({
                "Embedding": MODE_LABELS[mode],
                "Dim": config["embedding_dims"][mode],
                "Architecture": ARCH_LABELS[arch],
                "IMDB acc":   f"{m['imdb_test']['accuracy']*100:.2f}%",
                "IMDB loss":  f"{m['imdb_test']['loss']:.4f}",
                "TripAdvisor acc":  f"{m['tripadvisor']['accuracy']*100:.2f}%",
                "TripAdvisor loss": f"{m['tripadvisor']['loss']:.4f}",
            })
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True,
                     hide_index=True)

    # Bar chart for visual comparison (only if multiple modes available)
    if len(available_modes) >= 2:
        st.markdown("### Visual comparison")
        chart_rows = []
        for mode in available_modes:
            for arch in ("original", "improved"):
                if arch in results.get(mode, {}).get("models", {}):
                    m = results[mode]["models"][arch]
                    chart_rows.append({
                        "Model": f"{MODE_LABELS[mode]} · {ARCH_LABELS[arch]}",
                        "IMDB":        m["imdb_test"]["accuracy"] * 100,
                        "TripAdvisor": m["tripadvisor"]["accuracy"] * 100,
                    })
        chart_df = pd.DataFrame(chart_rows).set_index("Model")
        st.bar_chart(chart_df)

    st.markdown("---")
    st.markdown("### What's the difference between the embedding strategies?")
    st.markdown("""
| | **Trained from scratch** | **GloVe** | **Word2Vec** |
|---|---|---|---|
| **Source** | Random init, learned from 900 IMDB sentences | Pre-trained on 6B tokens (Wikipedia + Gigaword) | Pre-trained on 100B tokens (Google News) |
| **Dimensions** | 100 (configurable) | 100 (`glove.6B.100d`) | 300 |
| **Algorithm** | Backprop on the task | Global word co-occurrence factorization | Skip-gram with negative sampling |
| **Trainable** | Yes | No (frozen) | No (frozen) |
| **Strength** | Tailored exactly to the task | Captures global semantics, smaller | Rich syntactic+semantic, very large vocabulary |
| **Weakness** | Overfits on tiny data, no general knowledge | Smaller vocabulary (~400K) | 1.5 GB file, slow to load |
    """)

    st.markdown("### Expected pattern")
    st.markdown("""
- **On IMDB (in-domain):** Trained-from-scratch can match or beat pre-trained
  embeddings because it learns task-specific word meanings — but only if the
  dataset is big enough. With just 1000 sentences, pre-trained embeddings
  often help.
- **On TripAdvisor (cross-domain):** Pre-trained embeddings (especially Word2Vec)
  typically generalize **much better** because they encode general English
  semantics, not just IMDB-specific patterns. This is the headline story —
  pre-trained embeddings shine at cross-domain tasks.
- **GloVe vs Word2Vec:** Word2Vec is trained on a far larger corpus
  (100B tokens vs 6B) so it has better coverage of rare words, but GloVe is
  smaller and loads faster. On clean, common English text the two are usually
  close in performance.
    """)
