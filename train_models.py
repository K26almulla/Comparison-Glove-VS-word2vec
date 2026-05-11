"""
Train FOUR sentiment CNN models on IMDB labelled sentences,
then evaluate all four on TripAdvisor hotel reviews (cross-domain test):

    1. Original CNN  + GloVe 100d
    2. Improved CNN  + GloVe 100d
    3. Original CNN  + Word2Vec 300d (GoogleNews)
    4. Improved CNN  + Word2Vec 300d (GoogleNews)

Run this ONCE in Colab (the Word2Vec binary is 3.5 GB — keeping it in Drive
is fine). The script saves the trained .keras model files, the tokenizer,
config, and a results.json — these get uploaded to GitHub so the Streamlit
app can compare all four models WITHOUT needing the embedding binaries.

Expected files in the same folder as this script:
    imdb_labelled.tsv
    tripadvisor_hotel_reviews.csv
    glove.6B.100d.txt
    GoogleNews-vectors-negative300.bin
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
# Paths
# -----------------------------
HERE          = os.path.dirname(os.path.abspath(__file__))
IMDB_PATH     = os.path.join(HERE, "imdb_labelled.tsv")
TRIP_PATH     = os.path.join(HERE, "tripadvisor_hotel_reviews.csv")
GLOVE_PATH    = os.path.join(HERE, "glove.6B.100d.txt")
WORD2VEC_PATH = os.path.join(HERE, "GoogleNews-vectors-negative300.bin")
OUT_DIR       = os.path.join(HERE, "artifacts")
os.makedirs(OUT_DIR, exist_ok=True)

MAX_SEQUENCE_LENGTH_ORIG     = 50
MAX_SEQUENCE_LENGTH_IMPROVED = 100
GLOVE_DIM    = 100
W2V_DIM      = 300
MAX_VOCAB    = 10000
EPOCHS       = 15
BATCH_SIZE   = 32

# -----------------------------
# Preprocessing
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
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = [t for t in text.split() if t and t not in STOPWORDS and len(t) > 1]
    return " ".join(tokens)

# -----------------------------
# 1. Load datasets
# -----------------------------
print("Loading IMDB...")
imdb = pd.read_csv(IMDB_PATH, sep="\t", header=None, names=["Text", "Label"],
                   encoding="utf-8", engine="python", on_bad_lines="skip")
imdb["Text_Final"] = imdb["Text"].apply(clean_text)
print(f"  IMDB rows: {len(imdb)} | pos={imdb['Label'].sum()} neg={(imdb['Label']==0).sum()}")

print("Loading TripAdvisor...")
trip = pd.read_csv(TRIP_PATH)
trip = trip[trip["Rating"] != 3].copy()
trip["Label"] = (trip["Rating"] >= 4).astype(int)
trip["Text_Final"] = trip["Review"].apply(clean_text)
trip_sample = trip.sample(n=min(2000, len(trip)), random_state=42)
print(f"  TripAdvisor sample: {len(trip_sample)}")

# -----------------------------
# 2. Split + tokenize
# -----------------------------
data_train, data_test = train_test_split(
    imdb, test_size=0.1, random_state=42, stratify=imdb["Label"]
)

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
    arr[labels == 1, 0] = 1.0
    arr[labels == 0, 1] = 1.0
    return arr

y_train    = to_onehot(data_train["Label"].values)
y_test     = to_onehot(data_test["Label"].values)
y_trip     = to_onehot(trip_sample["Label"].values)
y_test_int = data_test["Label"].values
y_trip_int = trip_sample["Label"].values

# -----------------------------
# 3. Embedding matrices
# -----------------------------
def build_glove_matrix():
    print(f"Loading GloVe from {GLOVE_PATH}...")
    glove = {}
    with open(GLOVE_PATH, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip().split(" ")
            glove[parts[0]] = np.asarray(parts[1:], dtype="float32")
    mat = np.zeros((vocab_size, GLOVE_DIM), dtype="float32")
    hits = 0
    for word, idx in word_index.items():
        if idx >= vocab_size: continue
        v = glove.get(word)
        if v is not None:
            mat[idx] = v; hits += 1
    cov = hits / (vocab_size - 1) * 100
    print(f"  GloVe coverage: {hits}/{vocab_size-1} = {cov:.1f}%")
    return mat, cov

def build_word2vec_matrix():
    print(f"Loading Word2Vec from {WORD2VEC_PATH}...")
    from gensim.models import KeyedVectors
    wv = KeyedVectors.load_word2vec_format(WORD2VEC_PATH, binary=True)
    mat = np.zeros((vocab_size, W2V_DIM), dtype="float32")
    hits = 0
    for word, idx in word_index.items():
        if idx >= vocab_size: continue
        if word in wv:
            mat[idx] = wv[word]; hits += 1
    cov = hits / (vocab_size - 1) * 100
    print(f"  Word2Vec coverage: {hits}/{vocab_size-1} = {cov:.1f}%")
    del wv  # free 3.5 GB!
    return mat, cov

glove_matrix, glove_cov = build_glove_matrix()
w2v_matrix,   w2v_cov   = build_word2vec_matrix()

# -----------------------------
# 4. Model builders (parameterized by embedding matrix & dim)
# -----------------------------
def build_original_cnn(emb_matrix, emb_dim):
    seq_in = Input(shape=(MAX_SEQUENCE_LENGTH_ORIG,), dtype="int32")
    x = Embedding(input_dim=vocab_size, output_dim=emb_dim,
                  weights=[emb_matrix],
                  input_length=MAX_SEQUENCE_LENGTH_ORIG, trainable=False)(seq_in)
    x = Conv1D(200, 2, activation="relu")(x); x = MaxPooling1D()(x)
    x = Conv1D(200, 3, activation="relu")(x); x = MaxPooling1D()(x)
    x = Flatten()(x)
    x = Dropout(0.1)(x); x = Dense(128, activation="relu")(x)
    x = Dropout(0.1)(x); out = Dense(2, activation="sigmoid")(x)
    m = Model(seq_in, out)
    m.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return m

def build_improved_cnn(emb_matrix, emb_dim):
    seq_in = Input(shape=(MAX_SEQUENCE_LENGTH_IMPROVED,), dtype="int32")
    x = Embedding(input_dim=vocab_size, output_dim=emb_dim,
                  weights=[emb_matrix],
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
# 5. Train + evaluate one model
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

def train_one(label, builder, emb_matrix, emb_dim, maxlen, patience,
              train_X, test_X, trip_X, save_name):
    print(f"\n{'='*60}\nTraining: {label}\n{'='*60}")
    model = builder(emb_matrix, emb_dim)
    model.summary()
    es = EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True)
    hist = model.fit(train_X, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE,
                     validation_split=0.1, callbacks=[es], verbose=2)
    test_loss, test_acc = model.evaluate(test_X, y_test, verbose=0)
    trip_loss, trip_acc = model.evaluate(trip_X, y_trip, verbose=0)
    print(f"  IMDB:    acc={test_acc:.4f} loss={test_loss:.4f}")
    print(f"  TripAdv: acc={trip_acc:.4f} loss={trip_loss:.4f}")

    model.save(os.path.join(OUT_DIR, f"{save_name}.keras"))
    return {
        "label": label,
        "imdb_test": {"loss": float(test_loss), "accuracy": float(test_acc),
                      "details": detailed_eval(model, test_X, y_test_int)},
        "tripadvisor": {"loss": float(trip_loss), "accuracy": float(trip_acc),
                        "details": detailed_eval(model, trip_X, y_trip_int)},
        "history": {k: [float(v) for v in vs] for k, vs in hist.history.items()},
    }

# -----------------------------
# 6. Run all four
# -----------------------------
results = {}
results["glove_original"] = train_one(
    "Original CNN + GloVe 100d", build_original_cnn,
    glove_matrix, GLOVE_DIM, MAX_SEQUENCE_LENGTH_ORIG, 3,
    train_50, test_50, trip_50, "glove_original")

results["glove_improved"] = train_one(
    "Improved CNN + GloVe 100d", build_improved_cnn,
    glove_matrix, GLOVE_DIM, MAX_SEQUENCE_LENGTH_IMPROVED, 5,
    train_100, test_100, trip_100, "glove_improved")

results["word2vec_original"] = train_one(
    "Original CNN + Word2Vec 300d", build_original_cnn,
    w2v_matrix, W2V_DIM, MAX_SEQUENCE_LENGTH_ORIG, 3,
    train_50, test_50, trip_50, "word2vec_original")

results["word2vec_improved"] = train_one(
    "Improved CNN + Word2Vec 300d", build_improved_cnn,
    w2v_matrix, W2V_DIM, MAX_SEQUENCE_LENGTH_IMPROVED, 5,
    train_100, test_100, trip_100, "word2vec_improved")

# -----------------------------
# 7. Save tokenizer + config + results
# -----------------------------
print("\nSaving tokenizer and config...")
with open(os.path.join(OUT_DIR, "tokenizer.pkl"), "wb") as f:
    pickle.dump(tokenizer, f)

config = {
    "MAX_SEQUENCE_LENGTH_ORIG": MAX_SEQUENCE_LENGTH_ORIG,
    "MAX_SEQUENCE_LENGTH_IMPROVED": MAX_SEQUENCE_LENGTH_IMPROVED,
    "GLOVE_DIM": GLOVE_DIM,
    "W2V_DIM": W2V_DIM,
    "MAX_VOCAB": MAX_VOCAB,
    "vocab_size": vocab_size,
    "glove_coverage_pct": round(glove_cov, 2),
    "word2vec_coverage_pct": round(w2v_cov, 2),
    "models": {
        "glove_original":    {"file": "glove_original.keras",    "maxlen": MAX_SEQUENCE_LENGTH_ORIG,    "embedding": "GloVe 100d",   "arch": "Original CNN"},
        "glove_improved":    {"file": "glove_improved.keras",    "maxlen": MAX_SEQUENCE_LENGTH_IMPROVED,"embedding": "GloVe 100d",   "arch": "Improved CNN"},
        "word2vec_original": {"file": "word2vec_original.keras", "maxlen": MAX_SEQUENCE_LENGTH_ORIG,    "embedding": "Word2Vec 300d","arch": "Original CNN"},
        "word2vec_improved": {"file": "word2vec_improved.keras", "maxlen": MAX_SEQUENCE_LENGTH_IMPROVED,"embedding": "Word2Vec 300d","arch": "Improved CNN"},
    },
}
with open(os.path.join(OUT_DIR, "config.json"), "w") as f:
    json.dump(config, f, indent=2)

with open(os.path.join(OUT_DIR, "results.json"), "w") as f:
    json.dump(results, f, indent=2)

# -----------------------------
# 8. Summary table
# -----------------------------
print("\n" + "="*78)
print("SUMMARY — all four models")
print("="*78)
print(f"{'Model':<35}{'IMDB Acc':<12}{'IMDB Loss':<12}{'TripAdv Acc':<14}{'TripAdv Loss':<12}")
print("-"*78)
for k in ["glove_original", "glove_improved", "word2vec_original", "word2vec_improved"]:
    r = results[k]
    print(f"{r['label']:<35}"
          f"{r['imdb_test']['accuracy']*100:>8.2f}%   "
          f"{r['imdb_test']['loss']:>8.4f}    "
          f"{r['tripadvisor']['accuracy']*100:>8.2f}%      "
          f"{r['tripadvisor']['loss']:>8.4f}")
print("="*78)
print(f"\nAll artifacts saved to: {OUT_DIR}")
print("Files to push to GitHub (in artifacts/):")
for name in ["glove_original.keras", "glove_improved.keras",
             "word2vec_original.keras", "word2vec_improved.keras",
             "tokenizer.pkl", "config.json", "results.json"]:
    fp = os.path.join(OUT_DIR, name)
    if os.path.exists(fp):
        sz = os.path.getsize(fp) / 1e6
        print(f"  - {name}  ({sz:.1f} MB)")
print("\nDO NOT push glove.6B.100d.txt or GoogleNews-vectors-negative300.bin — they're huge and the app doesn't need them.")
