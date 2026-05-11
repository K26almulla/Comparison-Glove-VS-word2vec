# Sentiment Analysis CNN — Full Comparison Report

## Assignment summary
Build a Streamlit web app for sentiment analysis using the CNN from the lab,
improve the model meaningfully, compare the two, and test on a different
dataset. **Extension**: also compare embedding strategies (Scratch / GloVe /
Word2Vec) since the lab's GloVe notebook supports both pre-trained options.

## Datasets
- **Training:** `imdb_labelled.tsv` — 1000 IMDB movie review sentences,
  binary labels.
- **Cross-domain test:** `tripadvisor_hotel_reviews.csv` — hotel reviews
  with 1–5 star ratings. Ratings ≥4 → positive, ≤2 → negative, =3 dropped.
  2000 reviews sampled.

## Pipeline (matches the lab)
1. Lower-case + strip punctuation + remove common English stopwords.
   *(Lab uses spaCy; this app uses a regex tokenizer + small stopword list
   for reproducibility — same effect.)*
2. Keras `Tokenizer` fit on training set only (vocab capped at 10,000).
3. `pad_sequences` to fixed length.
4. `Embedding` layer — three modes supported:
   - **Scratch** (trainable, dim=100)
   - **GloVe** (`glove.6B.100d.txt`, frozen, dim=100)
   - **Word2Vec** (`GoogleNews-vectors-negative300.bin.gz`, frozen, dim=300)
5. CNN classifier with sigmoid output `[Pos, Neg]`.
6. `binary_crossentropy` + Adam + EarlyStopping (patience=5) on val_loss.

## Architectures

### Original CNN (from the lab)
```
Input(50)
  → Embedding(vocab, dim)
  → Conv1D(200, kernel=2, relu) → MaxPool
  → Conv1D(200, kernel=3, relu) → MaxPool
  → Flatten
  → Dropout(0.1) → Dense(128, relu)
  → Dropout(0.1) → Dense(2, sigmoid)
```

### Improved CNN
```
Input(100)                    # 2× longer context window
  → Embedding(vocab, dim)
  → [ Conv1D(64, k=2, relu, same)  → GlobalMaxPool ]    ┐
  → [ Conv1D(64, k=3, relu, same)  → GlobalMaxPool ]    │
  → [ Conv1D(64, k=4, relu, same)  → GlobalMaxPool ]    ├─ Concatenate (256-d)
  → [ Conv1D(64, k=5, relu, same)  → GlobalMaxPool ]    ┘
  → Dropout(0.4) → Dense(64, relu)
  → Dropout(0.3) → Dense(2, sigmoid)
```

### What changed and why
| Change | Original | Improved | Rationale |
|---|---|---|---|
| Sequence length | 50 | 100 | Captures more context for longer reviews |
| Conv structure | Sequential 2→3 stack | **Parallel branches** with kernels 2/3/4/5 | Sees multiple n-gram sizes at once |
| Pooling | `Flatten` (huge fixed vector) | `GlobalMaxPooling1D` per branch | Permutation-invariant, far fewer params |
| Dropout | 0.1 / 0.1 | **0.4 / 0.3** | Stronger regularization for small dataset |
| Dense head | 128 → 2 | 64 → 2 | Matches the leaner pooled representation |

## Embedding strategies compared

|  | **Scratch** | **GloVe** | **Word2Vec** |
|---|---|---|---|
| Source | Random init, learned from 900 IMDB sentences | Pre-trained on 6B tokens (Wikipedia + Gigaword) | Pre-trained on 100B tokens (Google News) |
| Dim | 100 | 100 | 300 |
| Algorithm | Backprop on the task | Global word co-occurrence factorization | Skip-gram with negative sampling |
| Trainable? | Yes | No (frozen) | No (frozen) |
| Size on disk | None (in-memory only) | ~330 MB | ~1.6 GB |
| Strength | Tailored exactly to the task | Captures global semantics | Rich syntactic+semantic, huge vocabulary |
| Weakness | Overfits on tiny data | Smaller vocabulary | Slow to load |

## Cross-domain evaluation strategy
Train on IMDB, test on **two** sets:
1. **In-domain:** held-out IMDB test split (10%, 75 sentences)
2. **Cross-domain:** 2000 TripAdvisor hotel reviews

The cross-domain test is the most informative: it shows whether the model
has learned **general sentiment signals** or just IMDB-specific patterns.

## Results — scratch embeddings (already trained, baseline)

| Architecture | IMDB acc | IMDB loss | TripAdvisor acc | TripAdvisor loss |
|---|---|---|---|---|
| Original CNN | ~70 % | ~0.90 | ~74 % | ~0.57 |
| **Improved CNN** | **~83 %** | **~0.41** | **~78 %** | **~0.49** |

Exact numbers vary slightly per run (random seed init); see
`artifacts/results.json` for the run that's currently saved.

## Results — GloVe + Word2Vec
**These will be filled in once you've downloaded the embedding files and
re-run training.** See `EMBEDDINGS_SETUP.md` for the download steps.

After re-running, the **Embedding comparison** tab in the app will show:
- A table with all (architecture × embedding) combinations
- A bar chart comparing IMDB vs TripAdvisor accuracy across all models
- Per-model loss numbers

### Expected pattern (based on prior literature)
- **On IMDB (in-domain):** All three embeddings perform similarly. Trained-
  from-scratch can sometimes win because it's tailored to IMDB-specific
  vocabulary, but pre-trained embeddings often help on small datasets
  (1000 sentences is tiny).
- **On TripAdvisor (cross-domain):** Pre-trained embeddings (especially
  Word2Vec) typically generalize **noticeably better** because they encode
  general English semantics instead of IMDB-specific co-occurrences.
- **GloVe vs Word2Vec:** Word2Vec usually has slightly better coverage of
  rare/technical words because of its much larger training corpus
  (100B tokens vs 6B). On clean review text the two are typically close.

## Why does cross-domain performance change?
1. **Vocabulary shift.** IMDB sentences talk about *plot, acting, scenes,
   characters*; TripAdvisor reviews talk about *rooms, staff, breakfast,
   parking*. Domain-specific words are often `<OOV>` for the IMDB-trained
   tokenizer. Pre-trained embeddings help because their vocabulary covers
   both domains.
2. **Length distribution shift.** IMDB sentences are short (one sentence each);
   TripAdvisor reviews are full paragraphs. The improved model's 100-token
   window helps but still truncates many reviews.
3. **Class imbalance in TripAdvisor.** ~81 % positive vs ~50 % in IMDB.
   Models trained on a balanced dataset under-predict the dominant class
   and thus get inflated raw accuracy if they tilt positive — macro-F1 is
   the more honest metric (visible in `artifacts/results.json`).
4. **Trainable vs frozen embeddings.** Scratch-mode embeddings learn
   IMDB-specific word vectors, which hurts on TripAdvisor. Frozen
   pre-trained embeddings keep generic English semantics intact.

## Files
- `train_models.py` — training pipeline; auto-detects available embeddings
- `app.py` — Streamlit web app
- `EMBEDDINGS_SETUP.md` — how to download GloVe and Word2Vec
- `artifacts/{original,improved}_cnn_{scratch,glove,word2vec}.keras` — trained models
- `artifacts/tokenizer.pkl` — fitted tokenizer
- `artifacts/config.json` — sequence lengths, vocab size, available modes
- `artifacts/results.json` — full metrics for every model

## How to reproduce
```bash
pip install tensorflow scikit-learn streamlit pandas numpy gensim
# (optional) download GloVe and/or Word2Vec — see EMBEDDINGS_SETUP.md
python3 train_models.py        # trains all detected embedding modes
streamlit run app.py            # launches the web app
```
