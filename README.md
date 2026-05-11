# Sentiment Analysis Web App (CNN)

Streamlit web app for the NLP lab bonus task. Trains CNN sentiment classifiers
with **multiple embedding strategies** (Scratch / GloVe / Word2Vec) and
**multiple architectures** (Original lab CNN / Improved CNN), compares them
all, and tests cross-domain on a different dataset.

## What's compared

The app presents **two comparisons**:

1. **Architecture comparison** — Original (lab) CNN vs Improved CNN
2. **Embedding comparison** — Trained-from-scratch vs GloVe vs Word2Vec

You can mix-and-match in the sidebar: pick any architecture + any available
embedding to run live predictions, then see all combinations side-by-side
in the comparison tabs.

## Quick start

```bash
# 1. Install dependencies
pip install tensorflow scikit-learn streamlit pandas numpy gensim

# 2. (Optional but recommended) Download pre-trained embeddings
#    See EMBEDDINGS_SETUP.md for full instructions
#    - glove.6B.100d.txt           (~330 MB after unzip)
#    - GoogleNews-vectors-negative300.bin.gz   (~1.6 GB)
#    Place them in this folder, next to train_models.py

# 3. Train models — auto-detects which embeddings are available
python3 train_models.py

# 4. Launch the app
streamlit run app.py
```

If you skip step 2, only the trained-from-scratch models will be trained,
and the app will still work — it just won't show the GloVe vs Word2Vec
comparison until you provide those files.

## Files

| File | Purpose |
|---|---|
| `train_models.py` | Trains models with whichever embeddings are present |
| `app.py` | Streamlit web app (4 tabs: prediction, batch, arch comparison, embedding comparison) |
| `EMBEDDINGS_SETUP.md` | Step-by-step guide to downloading GloVe & Word2Vec |
| `COMPARISON_REPORT.md` | Detailed model & embedding comparison writeup |
| `artifacts/*.keras` | Trained models (one per arch × embedding combination) |
| `artifacts/tokenizer.pkl` | Fitted Keras tokenizer |
| `artifacts/config.json` | Sequence lengths, vocab size, available modes |
| `artifacts/results.json` | Full metrics for every model |

## Datasets used
- **Training:** IMDB labelled sentences (the lab's dataset, 1000 sentences)
- **Cross-domain test:** TripAdvisor hotel reviews (20,491 reviews, 2000 sampled)

## App tabs

1. **🔤 Single sentence** — type a sentence, see prediction + confidence,
   plus how every other available model would have predicted the same input
2. **📂 CSV upload** — batch predict on a CSV; if you provide a ground-truth
   column, accuracy is computed live
3. **📈 Architecture comparison** — Original vs Improved (uses the currently
   selected embedding)
4. **🧬 Embedding comparison** — Scratch vs GloVe vs Word2Vec, with a bar chart
