"""
Train Original CNN (lab) and Improved CNN using GloVe 100d pretrained embeddings,
then evaluate on TripAdvisor hotel reviews (cross-domain test).

Run this ONCE (in Colab is fine) — it saves the trained .keras files,
the tokenizer, and the results JSON. The Streamlit app only loads these
artifacts; it does NOT need GloVe at deploy time.

Expected layout:
  ./imdb_labelled.tsv
  ./tripadvisor_hotel_reviews.csv
  ./glove.6B.100d.txt          # download once, ~330 MB
  ./artifacts/                 # created by this script
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
    GlobalMaxPooling1D, concatenate,
)
from tensorflow.keras.callbacks import EarlyStopping

np.random.seed(42)
tf.random.set_seed(42)

# -----------------------------
# Paths — edit if needed
# -----------------------------
HERE       = os.path.dirname(os.path.abspath(__file__))
IMDB_PATH  = os.path.join(HERE, "imdb_labelled.tsv")
TRIP_PATH  = os.path.join(HERE, "tripadvisor_hotel_reviews.csv")
GLOVE_PATH = os.path.join(HERE, "glove.6B.100d.txt")
OUT_DIR    = os.path.join(HERE, "artifacts")
os.makedirs(OUT_DIR, exist_ok=True)

MAX_SEQUENCE_LENGTH_ORIG     = 50
MAX_SEQUENCE_LENGTH_IMPROVED = 100
EMBEDDING_DIM = 100
MAX_VOCAB     = 10000
EPOCHS        = 15
BATCH_SIZE    = 32

# -----------------------------
# Preprocessing (same as before — regex tokenizer + small stopword list)
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
# 1. Load IMDB
# -----------------------------
print("Loading IMDB...")
imdb = pd.read_csv(IMDB_PATH, sep="\t", header=None, names=["Text", "Label"],
                   encoding="utf-8", engine="python", on_bad_lines="skip")
imdb["Text_Final"] = imdb["Text"].apply(clean_text)
print(f"IMDB rows: {len(imdb)} | pos={imdb['Label'].sum()} neg={(imdb['Label']==0).sum()}")

# -----------------------------
# 2. Load TripAdvisor (cross-domain test)
# -----------------------------
print("Loading TripAdvisor...")
trip = pd.read_csv(TRIP_PATH)
trip = trip[trip["Rating"] != 3].copy()
trip["Label"] = (trip["Rating"] >= 4).astype(int)
trip["Text_Final"] = trip["Review"].apply(clean_text)
trip_sample = trip.sample(n=min(2000, len(trip)), random_state=42)
print(f"TripAdvisor sample: {len(trip_sample)}")

# -----------------------------
# 3. Split
# -----------------------------
data_train, data_test = train_test_split(
    imdb, test_size=0.1, random_state=42, stratify=imdb["Label"]
)

# -----------------------------
# 4. Tokenize
# -----------------------------
tokenizer = Tokenizer(num_words=MAX_VOCAB, oov_token="<OOV>")
tokenizer.fit_on_texts(data_train["Text_Final"].tolist())
word_index = tokenizer.word_index
vocab_size = min(MAX_VOCAB, len(word_index) + 1)
print(f"Vocab size: {vocab_size}")

train_seq = tokenizer.texts_to_sequences(data_train["Text_Final"].tolist())
test_seq  = tokenizer.texts_to_sequences(data_test["Text_Final"].tolist())
trip_seq  = tokenizer.texts_to_sequences(trip_sample["Text_Final"].tolist())

train_50  = pad_sequences(train_seq, maxlen=MAX_SEQUENCE_LENGTH_ORIG)
test_50   = pad_sequences(test_seq,  maxlen=MAX_SEQUENCE_LENGTH_ORIG)
trip_50   = pad_sequences(trip_seq,  maxlen=MAX_SEQUENCE_LENGTH_ORIG)
train_100 = pad_sequences(train_seq, maxlen=MAX_SEQUENCE_LENGTH_IMPROVED)
test_100  = pad_sequences(test_seq,  maxlen=MAX_SEQUENCE_LENGTH_IMPROVED)
trip_100  = pad_sequences(trip_seq,  maxlen=MAX_SEQUENCE_LENGTH_IMPROVED)

def to_onehot(labels):
    arr = np.zeros((len(labels), 2), dtype=np.float32)
    arr[labels == 1, 0] = 1.0  # Pos
    arr[labels == 0, 1] = 1.0  # Neg
    return arr

y_train    = to_onehot(data_train["Label"].values)
y_test     = to_onehot(data_test["Label"].values)
y_trip     = to_onehot(trip_sample["Label"].values)
y_test_int = data_test["Label"].values
y_trip_int = trip_sample["Label"].values

# -----------------------------
# 5. Load GloVe 100d -> embedding matrix
# -----------------------------
print(f"Loading GloVe from {GLOVE_PATH} ...")
glove = {}
with open(GLOVE_PATH, "r", encoding="utf-8") as f:
    for line in f:
        parts = line.rstrip().split(" ")
        word = parts[0]
        vec  = np.asarray(parts[1:], dtype="float32")
        glove[word] = vec
print(f"GloVe vectors loaded: {len(glove)}")

embedding_matrix = np.zeros((vocab_size, EMBEDDING_DIM), dtype="float32")
hits = 0
for word, idx in word_index.items():
    if idx >= vocab_size:
        continue
    v = glove.get(word)
    if v is not None:
        embedding_matrix[idx] = v
        hits += 1
print(f"GloVe coverage: {hits}/{vocab_size-1} = {hits/(vocab_size-1)*100:.1f}%")

# Free GloVe dict — we have the matrix now
del glove

# -----------------------------
# 6. Models
# -----------------------------
def build_original_cnn():
    seq_in = Input(shape=(MAX_SEQUENCE_LENGTH_ORIG,), dtype="int32")
    x = Embedding(input_dim=vocab_size, output_dim=EMBEDDING_DIM,
                  weights=[embedding_matrix],
                  input_length=MAX_SEQUENCE_LENGTH_ORIG, trainable=False)(seq_in)
    x = Conv1D(200, 2, activation="relu")(x); x = MaxPooling1D()(x)
    x = Conv1D(200, 3, activation="relu")(x); x = MaxPooling1D()(x)
    x = Flatten()(x)
    x = Dropout(0.1)(x); x = Dense(128, activation="relu")(x)
    x = Dropout(0.1)(x); out = Dense(2, activation="sigmoid")(x)
    m = Model(seq_in, out)
    m.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return m

def build_improved_cnn():
    seq_in = Input(shape=(MAX_SEQUENCE_LENGTH_IMPROVED,), dtype="int32")
    x = Embedding(input_dim=vocab_size, output_dim=EMBEDDING_DIM,
                  weights=[embedding_matrix],
                  input_length=MAX_SEQUENCE_LENGTH_IMPROVED, trainable=False)(seq_in)
    branches = []
    for k in (2, 3, 4, 5):
        b = Conv1D(64, k, activation="relu", padding="same")(x)
        b = GlobalMaxPooling1D()(b)
        branches.append(b)
    x = concatenate(branches)
    x = Dropout(0.4)(x); x = Dense(64, activation="relu")(x)
    x = Dropout(0.3)(x); out = Dense(2, activation="sigmoid")(x)
    m = Model(seq_in, out)
    m.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return m

# -----------------------------
# 7. Train
# -----------------------------
print("\nTraining ORIGINAL CNN...")
orig = build_original_cnn()
orig.summary()
es1 = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
hist_orig = orig.fit(train_50, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE,
                     validation_split=0.1, callbacks=[es1], verbose=2)
orig_test_loss, orig_test_acc = orig.evaluate(test_50, y_test, verbose=0)
orig_trip_loss, orig_trip_acc = orig.evaluate(trip_50, y_trip, verbose=0)
print(f"Original IMDB acc: {orig_test_acc:.4f}  TripAdv acc: {orig_trip_acc:.4f}")

print("\nTraining IMPROVED CNN...")
improved = build_improved_cnn()
improved.summary()
es2 = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
hist_imp = improved.fit(train_100, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE,
                        validation_split=0.1, callbacks=[es2], verbose=2)
imp_test_loss, imp_test_acc = improved.evaluate(test_100, y_test, verbose=0)
imp_trip_loss, imp_trip_acc = improved.evaluate(trip_100, y_trip, verbose=0)
print(f"Improved IMDB acc: {imp_test_acc:.4f}  TripAdv acc: {imp_trip_acc:.4f}")

# -----------------------------
# 8. Detailed eval
# -----------------------------
def detailed_eval(model, X, y_int):
    probs = model.predict(X, verbose=0)
    preds = (probs[:, 0] > probs[:, 1]).astype(int)
    return {
        "accuracy": float(accuracy_score(y_int, preds)),
        "confusion_matrix": confusion_matrix(y_int, preds).tolist(),
        "report": classification_report(y_int, preds, target_names=["neg", "pos"],
                                        output_dict=True, zero_division=0),
    }

orig_imdb_d = detailed_eval(orig,     test_50,  y_test_int)
orig_trip_d = detailed_eval(orig,     trip_50,  y_trip_int)
imp_imdb_d  = detailed_eval(improved, test_100, y_test_int)
imp_trip_d  = detailed_eval(improved, trip_100, y_trip_int)

# -----------------------------
# 9. Save artifacts
# -----------------------------
print("\nSaving artifacts...")
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
    "embedding_source": "GloVe 6B 100d (pretrained, frozen)",
    "glove_coverage_pct": round(hits/(vocab_size-1)*100, 2),
}
with open(os.path.join(OUT_DIR, "config.json"), "w") as f:
    json.dump(config, f, indent=2)

results = {
    "original": {
        "imdb_test":   {"loss": float(orig_test_loss), "accuracy": float(orig_test_acc), "details": orig_imdb_d},
        "tripadvisor": {"loss": float(orig_trip_loss), "accuracy": float(orig_trip_acc), "details": orig_trip_d},
        "history": {k: [float(v) for v in vs] for k, vs in hist_orig.history.items()},
    },
    "improved": {
        "imdb_test":   {"loss": float(imp_test_loss), "accuracy": float(imp_test_acc), "details": imp_imdb_d},
        "tripadvisor": {"loss": float(imp_trip_loss), "accuracy": float(imp_trip_acc), "details": imp_trip_d},
        "history": {k: [float(v) for v in vs] for k, vs in hist_imp.history.items()},
    },
}
with open(os.path.join(OUT_DIR, "results.json"), "w") as f:
    json.dump(results, f, indent=2)

print(f"\nDone. Artifacts in: {OUT_DIR}")
print(f"  - original_cnn.keras  ({os.path.getsize(os.path.join(OUT_DIR,'original_cnn.keras'))/1e6:.1f} MB)")
print(f"  - improved_cnn.keras  ({os.path.getsize(os.path.join(OUT_DIR,'improved_cnn.keras'))/1e6:.1f} MB)")
print(f"  - tokenizer.pkl, config.json, results.json")
print("\nNote: do NOT commit glove.6B.100d.txt to GitHub — it's only needed for training.")
