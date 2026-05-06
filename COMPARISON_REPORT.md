# Sentiment Analysis CNN — Model Comparison Report

## Assignment summary
Build a Streamlit web app for sentiment analysis using the CNN from the lab,
improve the model meaningfully, compare the two, and test on a different
dataset.

## Datasets
- **Training dataset (matches the lab):** `imdb_labelled.tsv` — 1000 IMDB
  movie review sentences labelled positive (1) or negative (0).
- **Cross-domain test dataset:** `tripadvisor_hotel_reviews.csv` — 20,491
  hotel reviews with 1–5 star ratings. Converted to binary by mapping
  rating ≥ 4 → positive and rating ≤ 2 → negative (rating = 3 dropped).
  2000 reviews sampled for evaluation.

## Pipeline (reused from the lab)
1. Lower-case + strip punctuation + remove common English stopwords
   *(lab uses spaCy; we use a regex tokenizer + small stopword list for
   reproducibility — same effect)*
2. Keras `Tokenizer` fit on training set only (vocab capped at 10,000)
3. `pad_sequences` to fixed length
4. `Embedding` layer (trainable, 100-dim — GloVe/Word2Vec binaries are not
   bundled in this environment, so we let the embedding train from scratch)
5. CNN classifier with sigmoid output `[Pos, Neg]`
6. `binary_crossentropy` loss, Adam optimizer, EarlyStopping on val_loss

## Architectures

### Original CNN (from the lab)
```
Input(50)
  → Embedding(vocab, 100, trainable=True)
  → Conv1D(200, kernel=2, relu) → MaxPool
  → Conv1D(200, kernel=3, relu) → MaxPool
  → Flatten
  → Dropout(0.1) → Dense(128, relu)
  → Dropout(0.1) → Dense(2, sigmoid)
```

### Improved CNN
```
Input(100)                    # 2× longer context window
  → Embedding(vocab, 100, trainable=True)
  → [ Conv1D(64, k=2, relu, same)  → GlobalMaxPool ]    ┐
  → [ Conv1D(64, k=3, relu, same)  → GlobalMaxPool ]    │
  → [ Conv1D(64, k=4, relu, same)  → GlobalMaxPool ]    ├─ Concatenate (256-d)
  → [ Conv1D(64, k=5, relu, same)  → GlobalMaxPool ]    ┘
  → Dropout(0.4) → Dense(64, relu)
  → Dropout(0.3) → Dense(2, sigmoid)
```

### What changed and why (more than just epochs)
| Change | Original | Improved | Rationale |
|---|---|---|---|
| Sequence length | 50 | 100 | Captures more context, especially for longer reviews in TripAdvisor |
| Conv structure | Sequential 2→3 stack | **Parallel branches** with kernels 2/3/4/5 | Sees multiple n-gram sizes at once instead of losing info through pooling |
| Pooling | `Flatten` (huge fixed vector) | `GlobalMaxPooling1D` per branch | Permutation-invariant, far fewer dense-layer parameters → less overfitting |
| Dropout | 0.1 / 0.1 | **0.4 / 0.3** | Much stronger regularization for the small (1000-sentence) IMDB set |
| Dense head | 128 → 2 | 64 → 2 | Smaller head matches the leaner pooled representation |

## Results

### IMDB held-out test (10 % split, 75 sentences)
| Model | Accuracy | Loss |
|---|---|---|
| Original CNN (lab) | 70.67 % | 0.9020 |
| **Improved CNN** | **84.00 %** | **0.3971** |

### TripAdvisor cross-domain test (2000 reviews)
| Model | Accuracy | Loss |
|---|---|---|
| Original CNN (lab) | 60.05 % | 1.2040 |
| **Improved CNN** | **81.55 %** | **0.4339** |

## Which one is better?
The **Improved CNN** wins on **both** evaluations:
- On the in-domain IMDB test it gains +13.3 percentage points.
- On the cross-domain TripAdvisor test it gains +21.5 percentage points
  *and* slashes loss by nearly 3×.

The original CNN's heavy reliance on `Flatten` produces a huge first-dense
vector (`floor(((50-2+1)/2 - 3+1)/2) × 200 ≈ 2200` features) that overfits
the 900 training sentences badly. The improved CNN's GlobalMaxPooling
yields exactly 256 features — orders of magnitude leaner, paired with
much stronger dropout, so it learns more general patterns.

## Cross-domain analysis — does performance change?
**Yes, both models drop on the cross-domain test, but in different ways:**

- The original CNN drops from 70.7 % → 60.1 % (-10.6 pp). Its predictions
  on TripAdvisor are noisy in both directions — it has not learned
  robust sentiment signals.
- The improved CNN drops only slightly from 84.0 % → 81.5 % (-2.5 pp).

**Why the drop?**
1. **Domain shift in vocabulary.** IMDB sentences talk about *plot,
   acting, scenes, characters*; TripAdvisor reviews talk about *rooms,
   staff, breakfast, parking*. Domain-specific words are often `<OOV>`
   for the IMDB-trained tokenizer.
2. **Length distribution shift.** IMDB sentences are short (one sentence
   each); TripAdvisor reviews are full paragraphs. The improved model's
   100-token window helps but still truncates many reviews.
3. **Class imbalance in TripAdvisor.** ~81 % positive vs ~50 % in IMDB.
   Models trained on a balanced dataset under-predict the dominant class
   and thus get inflated accuracy if they tilt positive — look at the
   per-class metrics in the training log: negative-class recall is
   weak for both models on TripAdvisor, which is the honest story.
4. **Trainable embeddings learned IMDB-specific word vectors.** Using
   pre-trained GloVe/Word2Vec embeddings (as in the lab) would likely
   reduce the domain-shift gap because the embeddings would already
   encode general English semantics.

## Files in this submission
- `train_models.py` — full training pipeline (loads IMDB, trains both
  models, evaluates on TripAdvisor, saves all artifacts)
- `app.py` — Streamlit web app (single-sentence prediction, CSV upload,
  model comparison tab)
- `artifacts/original_cnn.keras` — trained original lab model
- `artifacts/improved_cnn.keras` — trained improved model
- `artifacts/tokenizer.pkl` — fitted Keras tokenizer
- `artifacts/config.json` — sequence lengths, vocab size, etc.
- `artifacts/results.json` — full evaluation metrics including confusion
  matrices and per-class precision/recall/F1
- `README.md` — how to run

## How to reproduce
```bash
pip install tensorflow scikit-learn streamlit pandas
python3 train_models.py        # trains both models, ~30s on CPU
streamlit run app.py            # launches the web app
```
