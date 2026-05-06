# Sentiment Analysis Web App (CNN)

Streamlit web app for the NLP lab bonus task. Trains two CNN sentiment
classifiers (original lab architecture + an improved version), compares them,
and tests on a different dataset.

## Quick start

```bash
# 1. Install dependencies
pip install tensorflow scikit-learn streamlit pandas numpy

# 2. Make sure the datasets are at these paths (or edit train_models.py):
#    /mnt/user-data/uploads/imdb_labelled__1___1_.tsv
#    /mnt/user-data/uploads/tripadvisor_hotel_reviews.csv

# 3. Train both models (~30 seconds on CPU)
python3 train_models.py

# 4. Launch the web app
streamlit run app.py
```

## What the app does
- **Single-sentence prediction** with confidence score
- **CSV upload** for batch prediction (with optional ground-truth column
  for accuracy reporting)
- **Model selector** — switch between Original (lab) CNN and Improved CNN
  in the sidebar
- **Comparison tab** — side-by-side architecture and metrics

## Headline results

| Model | IMDB Acc | IMDB Loss | TripAdvisor Acc | TripAdvisor Loss |
|---|---|---|---|---|
| Original CNN (lab) | 70.67 % | 0.9020 | 60.05 % | 1.2040 |
| **Improved CNN** | **84.00 %** | **0.3971** | **81.55 %** | **0.4339** |

See `COMPARISON_REPORT.md` for the full analysis.

## Files
| File | Purpose |
|---|---|
| `train_models.py` | Trains both models on IMDB, evaluates on TripAdvisor, saves artifacts |
| `app.py` | Streamlit web app |
| `artifacts/original_cnn.keras` | Trained lab model |
| `artifacts/improved_cnn.keras` | Trained improved model |
| `artifacts/tokenizer.pkl` | Fitted Keras tokenizer |
| `artifacts/config.json` | Sequence lengths, vocab size, embedding dim |
| `artifacts/results.json` | Full per-class metrics + training history |
| `COMPARISON_REPORT.md` | Detailed model comparison & cross-domain analysis |
