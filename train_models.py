"""
Train both Original CNN (from lab) and Improved CNN on IMDB labelled sentences.
Then test both models on TripAdvisor Hotel Reviews (a different dataset / domain)
to study cross-domain generalization.

This script reuses the lab's pipeline:
- Tokenization + padding
- Embedding layer (trainable, since GloVe/Word2Vec binaries aren't bundled)
- CNN architecture for binary sentiment classification
"""
import os
import json
import pickle
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Embedding, Conv1D, MaxPooling1D, Flatten, Dropout, Dense,
    GlobalMaxPooling1D, BatchNormalization, concatenate,
)
from tensorflow.keras.callbacks import EarlyStopping

# Reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# -----------------------------
# Paths & config
# -----------------------------
IMDB_PATH = "/mnt/user-data/uploads/imdb_labelled__1___1_.tsv"
TRIP_PATH = "/mnt/user-data/uploads/tripadvisor_hotel_reviews.csv"
OUT_DIR   = "/home/claude/sentiment_app/artifacts"
os.makedirs(OUT_DIR, exist_ok=True)

MAX_SEQUENCE_LENGTH_ORIG     = 50    # matches the lab
MAX_SEQUENCE_LENGTH_IMPROVED = 100   # improved: longer context window
EMBEDDING_DIM = 100
MAX_VOCAB     = 10000
EPOCHS        = 15
BATCH_SIZE    = 32

# -----------------------------
# Lightweight preprocessing
# (lab uses spaCy; we use a regex tokenizer + small stopword list for the
#  same effect: lower / strip punctuation / drop common stopwords. Keeps
#  the script dependency-light and reproducible.)
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
# 1. Load IMDB (training dataset, matches the lab)
# -----------------------------
print("=" * 60)
print("Loading IMDB labelled sentences (training data)...")
print("=" * 60)
imdb = pd.read_csv(IMDB_PATH, sep="\t", header=None,
                   names=["Text", "Label"], encoding="utf-8",
                   engine="python", on_bad_lines="skip")
print(f"IMDB rows: {len(imdb)}")
print(f"IMDB class balance: pos={imdb['Label'].sum()}, "
      f"neg={(imdb['Label']==0).sum()}")

imdb["Text_Final"] = imdb["Text"].apply(clean_text)

# -----------------------------
# 2. Load TripAdvisor (cross-domain test dataset)
# -----------------------------
print("\nLoading TripAdvisor hotel reviews (cross-domain test data)...")
trip = pd.read_csv(TRIP_PATH)
print(f"TripAdvisor rows: {len(trip)}")
# Convert 1–5 ratings to binary: >=4 -> positive, <=2 -> negative, drop 3
trip = trip[trip["Rating"] != 3].copy()
trip["Label"] = (trip["Rating"] >= 4).astype(int)
trip["Text_Final"] = trip["Review"].apply(clean_text)
print(f"After dropping neutral (rating=3): {len(trip)} rows")
print(f"TripAdvisor class balance: pos={trip['Label'].sum()}, "
      f"neg={(trip['Label']==0).sum()}")

# Subsample TripAdvisor for a manageable cross-domain test set
trip_sample = trip.sample(n=min(2000, len(trip)), random_state=42)
print(f"Sampled {len(trip_sample)} TripAdvisor reviews for testing")

# -----------------------------
# 3. Train/test split on IMDB
# -----------------------------
data_train, data_test = train_test_split(
    imdb, test_size=0.1, random_state=42, stratify=imdb["Label"]
)
print(f"\nIMDB train: {len(data_train)} | IMDB test: {len(data_test)}")

# -----------------------------
# 4. Tokenize on IMDB training set only
# -----------------------------
tokenizer = Tokenizer(num_words=MAX_VOCAB, oov_token="<OOV>")
tokenizer.fit_on_texts(data_train["Text_Final"].tolist())
word_index = tokenizer.word_index
print(f"Vocabulary size (capped at {MAX_VOCAB}): "
      f"{min(len(word_index), MAX_VOCAB)}")

train_sequences = tokenizer.texts_to_sequences(data_train["Text_Final"].tolist())
test_sequences  = tokenizer.texts_to_sequences(data_test["Text_Final"].tolist())
trip_sequences  = tokenizer.texts_to_sequences(trip_sample["Text_Final"].tolist())

# Pad for both length configurations
train_data_50  = pad_sequences(train_sequences, maxlen=MAX_SEQUENCE_LENGTH_ORIG)
test_data_50   = pad_sequences(test_sequences,  maxlen=MAX_SEQUENCE_LENGTH_ORIG)
trip_data_50   = pad_sequences(trip_sequences,  maxlen=MAX_SEQUENCE_LENGTH_ORIG)

train_data_100 = pad_sequences(train_sequences, maxlen=MAX_SEQUENCE_LENGTH_IMPROVED)
test_data_100  = pad_sequences(test_sequences,  maxlen=MAX_SEQUENCE_LENGTH_IMPROVED)
trip_data_100  = pad_sequences(trip_sequences,  maxlen=MAX_SEQUENCE_LENGTH_IMPROVED)

# Labels in [Pos, Neg] one-hot format (matching the lab)
def to_onehot(labels):
    arr = np.zeros((len(labels), 2), dtype=np.float32)
    arr[labels == 1, 0] = 1.0  # Pos
    arr[labels == 0, 1] = 1.0  # Neg
    return arr

y_train     = to_onehot(data_train["Label"].values)
y_test      = to_onehot(data_test["Label"].values)
y_trip      = to_onehot(trip_sample["Label"].values)
y_test_int  = data_test["Label"].values
y_trip_int  = trip_sample["Label"].values

vocab_size = min(MAX_VOCAB, len(word_index) + 1)

# -----------------------------
# 5. Original CNN — straight from the lab
# -----------------------------
def build_original_cnn():
    seq_in = Input(shape=(MAX_SEQUENCE_LENGTH_ORIG,), dtype="int32")
    x = Embedding(input_dim=vocab_size, output_dim=EMBEDDING_DIM,
                  input_length=MAX_SEQUENCE_LENGTH_ORIG, trainable=True)(seq_in)
    x = Conv1D(200, 2, activation="relu")(x)
    x = MaxPooling1D()(x)
    x = Conv1D(200, 3, activation="relu")(x)
    x = MaxPooling1D()(x)
    x = Flatten()(x)
    x = Dropout(0.1)(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.1)(x)
    out = Dense(2, activation="sigmoid")(x)
    m = Model(seq_in, out)
    m.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return m

# -----------------------------
# 6. Improved CNN — meaningful changes vs original:
#    1) Sequence length 50 -> 100 (captures more context for longer reviews)
#    2) Multi-branch Conv1D with kernel sizes 2/3/4/5 (richer n-gram features
#       than the original's sequential 2->3 stack)
#    3) GlobalMaxPooling1D instead of Flatten (size-invariant, far fewer
#       parameters in the dense head, less prone to overfitting)
#    4) Stronger Dropout (0.4 / 0.3 vs original 0.1) to combat overfitting
#       on the small IMDB dataset
#    5) Smaller per-branch filter count (64 vs 200) — total ~256 filter
#       channels after concat, comparable capacity but better organized
# -----------------------------
def build_improved_cnn():
    seq_in = Input(shape=(MAX_SEQUENCE_LENGTH_IMPROVED,), dtype="int32")
    x = Embedding(input_dim=vocab_size, output_dim=EMBEDDING_DIM,
                  input_length=MAX_SEQUENCE_LENGTH_IMPROVED, trainable=True)(seq_in)

    # Multi-branch CNN — different kernel sizes capture different n-gram patterns
    branches = []
    for k in (2, 3, 4, 5):
        b = Conv1D(64, k, activation="relu", padding="same")(x)
        b = GlobalMaxPooling1D()(b)
        branches.append(b)
    x = concatenate(branches)

    x = Dropout(0.4)(x)
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.3)(x)
    out = Dense(2, activation="sigmoid")(x)
    m = Model(seq_in, out)
    m.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return m

# -----------------------------
# 7. Train original
# -----------------------------
print("\n" + "=" * 60)
print("Training ORIGINAL CNN (lab architecture)...")
print("=" * 60)
orig = build_original_cnn()
orig.summary()
es = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)

hist_orig = orig.fit(
    train_data_50, y_train,
    epochs=EPOCHS, batch_size=BATCH_SIZE,
    validation_split=0.1, callbacks=[es], verbose=2,
)

orig_test_loss, orig_test_acc = orig.evaluate(test_data_50, y_test, verbose=0)
print(f"\nOriginal CNN  IMDB test loss: {orig_test_loss:.4f}  "
      f"acc: {orig_test_acc:.4f}")

# Cross-domain eval on TripAdvisor
orig_trip_loss, orig_trip_acc = orig.evaluate(trip_data_50, y_trip, verbose=0)
print(f"Original CNN  TripAdvisor loss: {orig_trip_loss:.4f}  "
      f"acc: {orig_trip_acc:.4f}")

# -----------------------------
# 8. Train improved
# -----------------------------
print("\n" + "=" * 60)
print("Training IMPROVED CNN...")
print("=" * 60)
improved = build_improved_cnn()
improved.summary()
es2 = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

hist_imp = improved.fit(
    train_data_100, y_train,
    epochs=EPOCHS, batch_size=BATCH_SIZE,
    validation_split=0.1, callbacks=[es2], verbose=2,
)

imp_test_loss, imp_test_acc = improved.evaluate(test_data_100, y_test, verbose=0)
print(f"\nImproved CNN  IMDB test loss: {imp_test_loss:.4f}  "
      f"acc: {imp_test_acc:.4f}")

imp_trip_loss, imp_trip_acc = improved.evaluate(trip_data_100, y_trip, verbose=0)
print(f"Improved CNN  TripAdvisor loss: {imp_trip_loss:.4f}  "
      f"acc: {imp_trip_acc:.4f}")

# -----------------------------
# 9. Detailed per-class metrics
# -----------------------------
def detailed_eval(model, X, y_int, name):
    probs = model.predict(X, verbose=0)
    preds = (probs[:, 0] > probs[:, 1]).astype(int)  # Pos prob > Neg prob
    acc = accuracy_score(y_int, preds)
    cm  = confusion_matrix(y_int, preds).tolist()
    rep = classification_report(y_int, preds, target_names=["neg", "pos"],
                                output_dict=True, zero_division=0)
    print(f"\n--- {name} ---")
    print(f"Accuracy: {acc:.4f}")
    print(f"Confusion matrix [[TN,FP],[FN,TP]]: {cm}")
    print(classification_report(y_int, preds, target_names=["neg", "pos"],
                                zero_division=0))
    return {"accuracy": acc, "confusion_matrix": cm, "report": rep}

print("\n" + "=" * 60)
print("DETAILED EVALUATION")
print("=" * 60)
orig_imdb_metrics = detailed_eval(orig, test_data_50, y_test_int,
                                  "Original CNN — IMDB test")
orig_trip_metrics = detailed_eval(orig, trip_data_50, y_trip_int,
                                  "Original CNN — TripAdvisor (cross-domain)")
imp_imdb_metrics  = detailed_eval(improved, test_data_100, y_test_int,
                                  "Improved CNN — IMDB test")
imp_trip_metrics  = detailed_eval(improved, trip_data_100, y_trip_int,
                                  "Improved CNN — TripAdvisor (cross-domain)")

# -----------------------------
# 10. Save artifacts for the Streamlit app
# -----------------------------
print("\n" + "=" * 60)
print("Saving models and artifacts...")
print("=" * 60)
orig.save(os.path.join(OUT_DIR, "original_cnn.keras"))
improved.save(os.path.join(OUT_DIR, "improved_cnn.keras"))

with open(os.path.join(OUT_DIR, "tokenizer.pkl"), "wb") as f:
    pickle.dump(tokenizer, f)

config = {
    "MAX_SEQUENCE_LENGTH_ORIG": MAX_SEQUENCE_LENGTH_ORIG,
    "MAX_SEQUENCE_LENGTH_IMPROVED": MAX_SEQUENCE_LENGTH_IMPROVED,
    "EMBEDDING_DIM": EMBEDDING_DIM,
    "MAX_VOCAB": MAX_VOCAB,
    "vocab_size": vocab_size,
}
with open(os.path.join(OUT_DIR, "config.json"), "w") as f:
    json.dump(config, f, indent=2)

results = {
    "original": {
        "imdb_test":   {"loss": float(orig_test_loss),
                        "accuracy": float(orig_test_acc),
                        "details": orig_imdb_metrics},
        "tripadvisor": {"loss": float(orig_trip_loss),
                        "accuracy": float(orig_trip_acc),
                        "details": orig_trip_metrics},
        "history": {k: [float(v) for v in vs]
                    for k, vs in hist_orig.history.items()},
    },
    "improved": {
        "imdb_test":   {"loss": float(imp_test_loss),
                        "accuracy": float(imp_test_acc),
                        "details": imp_imdb_metrics},
        "tripadvisor": {"loss": float(imp_trip_loss),
                        "accuracy": float(imp_trip_acc),
                        "details": imp_trip_metrics},
        "history": {k: [float(v) for v in vs]
                    for k, vs in hist_imp.history.items()},
    },
}
with open(os.path.join(OUT_DIR, "results.json"), "w") as f:
    json.dump(results, f, indent=2)

print("Saved:")
for fname in ("original_cnn.keras", "improved_cnn.keras", "tokenizer.pkl",
              "config.json", "results.json"):
    print(f"  - {os.path.join(OUT_DIR, fname)}")

print("\n" + "=" * 60)
print("SUMMARY TABLE")
print("=" * 60)
print(f"{'Model':<22}{'IMDB Acc':<12}{'IMDB Loss':<12}"
      f"{'TripAdv Acc':<14}{'TripAdv Loss':<12}")
print("-" * 72)
print(f"{'Original CNN':<22}{orig_test_acc:<12.4f}{orig_test_loss:<12.4f}"
      f"{orig_trip_acc:<14.4f}{orig_trip_loss:<12.4f}")
print(f"{'Improved CNN':<22}{imp_test_acc:<12.4f}{imp_test_loss:<12.4f}"
      f"{imp_trip_acc:<14.4f}{imp_trip_loss:<12.4f}")
print("=" * 60)
print("Done.")
