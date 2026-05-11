"""
Train CNN sentiment classifiers with different embedding strategies and compare.

Embedding modes supported:
  - 'scratch'   : trainable embeddings learned from data (no pre-trained file needed)
  - 'glove'     : pre-trained GloVe vectors (e.g. glove.6B.100d.txt), frozen
  - 'word2vec'  : pre-trained Word2Vec (GoogleNews-vectors-negative300.bin), frozen

For each available embedding, both the Original (lab) CNN and the Improved CNN
are trained on the IMDB labelled sentences and then evaluated on:
  1. IMDB test split (in-domain)
  2. TripAdvisor hotel reviews (cross-domain)

Place the pre-trained files (if you have them) next to this script:
  ./glove.6B.100d.txt
  ./GoogleNews-vectors-negative300.bin
The script automatically detects which files are present and trains accordingly.
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

# Reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# -----------------------------
# Paths & config
# -----------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

IMDB_PATH = os.environ.get(
    "IMDB_PATH",
    "/mnt/user-data/uploads/imdb_labelled__1___1_.tsv",
)
TRIP_PATH = os.environ.get(
    "TRIP_PATH",
    "/mnt/user-data/uploads/tripadvisor_hotel_reviews.csv",
)
# Fallback to local copies if absolute paths don't exist
if not os.path.exists(IMDB_PATH):
    local = os.path.join(SCRIPT_DIR, "imdb_labelled.tsv")
    if os.path.exists(local):
        IMDB_PATH = local
if not os.path.exists(TRIP_PATH):
    local = os.path.join(SCRIPT_DIR, "tripadvisor_hotel_reviews.csv")
    if os.path.exists(local):
        TRIP_PATH = local

GLOVE_PATH    = os.path.join(SCRIPT_DIR, "glove.6B.100d.txt")
WORD2VEC_PATH = os.path.join(SCRIPT_DIR, "GoogleNews-vectors-negative300.bin")

OUT_DIR = os.path.join(SCRIPT_DIR, "artifacts")
os.makedirs(OUT_DIR, exist_ok=True)

MAX_SEQUENCE_LENGTH_ORIG     = 50
MAX_SEQUENCE_LENGTH_IMPROVED = 100
MAX_VOCAB                    = 10000
EPOCHS                       = 15
BATCH_SIZE                   = 32

# -----------------------------
# Lightweight preprocessing — same regex tokenizer for reproducibility
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
# Embedding loaders
# -----------------------------
def load_glove(path):
    print(f"Loading GloVe from {path}...")
    glove = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip().split(" ")
            word = parts[0]
            vec  = np.asarray(parts[1:], dtype="float32")
            glove[word] = vec
    dim = len(next(iter(glove.values())))
    print(f"  Loaded {len(glove)} GloVe vectors, dim={dim}")
    return glove, dim

def load_word2vec(path):
    print(f"Loading Word2Vec from {path}...")
    try:
        from gensim.models import KeyedVectors
    except ImportError:
        print("  gensim not installed; skipping Word2Vec. "
              "Install with: pip install gensim")
        return None, None
    wv = KeyedVectors.load_word2vec_format(path, binary=True)
    dim = wv.vector_size
    print(f"  Loaded {len(wv.key_to_index)} Word2Vec vectors, dim={dim}")
    return wv, dim

def build_embedding_matrix(word_index, vectors, dim, mode):
    vocab_size = min(MAX_VOCAB, len(word_index) + 1)
    matrix = np.zeros((vocab_size, dim), dtype="float32")
    found = 0
    for word, idx in word_index.items():
        if idx >= vocab_size:
            continue
        if word in vectors:
            matrix[idx] = vectors[word]
            found += 1
    print(f"  Embedding coverage: {found}/{vocab_size} "
          f"({100*found/vocab_size:.1f}%)")
    return matrix, vocab_size

# -----------------------------
# Model builders
# -----------------------------
def build_original_cnn(vocab_size, embedding_dim, embedding_weights=None):
    seq_in = Input(shape=(MAX_SEQUENCE_LENGTH_ORIG,), dtype="int32")
    if embedding_weights is not None:
        x = Embedding(
            input_dim=vocab_size, output_dim=embedding_dim,
            weights=[embedding_weights],
            input_length=MAX_SEQUENCE_LENGTH_ORIG, trainable=False,
        )(seq_in)
    else:
        x = Embedding(
            input_dim=vocab_size, output_dim=embedding_dim,
            input_length=MAX_SEQUENCE_LENGTH_ORIG, trainable=True,
        )(seq_in)
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

def build_improved_cnn(vocab_size, embedding_dim, embedding_weights=None):
    seq_in = Input(shape=(MAX_SEQUENCE_LENGTH_IMPROVED,), dtype="int32")
    if embedding_weights is not None:
        x = Embedding(
            input_dim=vocab_size, output_dim=embedding_dim,
            weights=[embedding_weights],
            input_length=MAX_SEQUENCE_LENGTH_IMPROVED, trainable=False,
        )(seq_in)
    else:
        x = Embedding(
            input_dim=vocab_size, output_dim=embedding_dim,
            input_length=MAX_SEQUENCE_LENGTH_IMPROVED, trainable=True,
        )(seq_in)
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
# Eval helpers
# -----------------------------
def to_onehot(labels):
    arr = np.zeros((len(labels), 2), dtype=np.float32)
    arr[labels == 1, 0] = 1.0
    arr[labels == 0, 1] = 1.0
    return arr

def detailed_eval(model, X, y_int):
    probs = model.predict(X, verbose=0)
    preds = (probs[:, 0] > probs[:, 1]).astype(int)
    acc = accuracy_score(y_int, preds)
    cm  = confusion_matrix(y_int, preds).tolist()
    rep = classification_report(y_int, preds, target_names=["neg", "pos"],
                                output_dict=True, zero_division=0)
    return {"accuracy": float(acc), "confusion_matrix": cm, "report": rep}

# =============================
# MAIN
# =============================
def main():
    print("=" * 60)
    print("Loading IMDB labelled sentences...")
    print("=" * 60)
    imdb = pd.read_csv(IMDB_PATH, sep="\t", header=None,
                       names=["Text", "Label"], encoding="utf-8",
                       engine="python", on_bad_lines="skip")
    imdb["Text_Final"] = imdb["Text"].apply(clean_text)
    print(f"IMDB rows: {len(imdb)}")

    print("\nLoading TripAdvisor (cross-domain test)...")
    trip = pd.read_csv(TRIP_PATH)
    trip = trip[trip["Rating"] != 3].copy()
    trip["Label"] = (trip["Rating"] >= 4).astype(int)
    trip["Text_Final"] = trip["Review"].apply(clean_text)
    trip_sample = trip.sample(n=min(2000, len(trip)), random_state=42)
    print(f"TripAdvisor rows: {len(trip)}; sampled {len(trip_sample)}")

    data_train, data_test = train_test_split(
        imdb, test_size=0.1, random_state=42, stratify=imdb["Label"],
    )

    tokenizer = Tokenizer(num_words=MAX_VOCAB, oov_token="<OOV>")
    tokenizer.fit_on_texts(data_train["Text_Final"].tolist())
    word_index = tokenizer.word_index
    print(f"\nVocabulary size (capped at {MAX_VOCAB}): "
          f"{min(len(word_index), MAX_VOCAB)}")

    train_seq = tokenizer.texts_to_sequences(data_train["Text_Final"].tolist())
    test_seq  = tokenizer.texts_to_sequences(data_test["Text_Final"].tolist())
    trip_seq  = tokenizer.texts_to_sequences(trip_sample["Text_Final"].tolist())

    train50  = pad_sequences(train_seq, maxlen=MAX_SEQUENCE_LENGTH_ORIG)
    test50   = pad_sequences(test_seq,  maxlen=MAX_SEQUENCE_LENGTH_ORIG)
    trip50   = pad_sequences(trip_seq,  maxlen=MAX_SEQUENCE_LENGTH_ORIG)
    train100 = pad_sequences(train_seq, maxlen=MAX_SEQUENCE_LENGTH_IMPROVED)
    test100  = pad_sequences(test_seq,  maxlen=MAX_SEQUENCE_LENGTH_IMPROVED)
    trip100  = pad_sequences(trip_seq,  maxlen=MAX_SEQUENCE_LENGTH_IMPROVED)

    y_train = to_onehot(data_train["Label"].values)
    y_test  = to_onehot(data_test["Label"].values)
    y_trip  = to_onehot(trip_sample["Label"].values)
    y_test_int = data_test["Label"].values
    y_trip_int = trip_sample["Label"].values

    # Decide which embedding modes to run
    modes = [("scratch", 100, None)]
    if os.path.exists(GLOVE_PATH):
        glove, gdim = load_glove(GLOVE_PATH)
        modes.append(("glove", gdim, glove))
    else:
        print(f"\n[skip] GloVe file not found at {GLOVE_PATH}")
    if os.path.exists(WORD2VEC_PATH):
        wv, wdim = load_word2vec(WORD2VEC_PATH)
        if wv is not None:
            modes.append(("word2vec", wdim, wv))
    else:
        print(f"[skip] Word2Vec file not found at {WORD2VEC_PATH}")

    print(f"\nTraining with embedding modes: {[m[0] for m in modes]}")

    results = {}
    for mode_name, dim, vectors in modes:
        print("\n" + "=" * 60)
        print(f"EMBEDDING MODE: {mode_name.upper()}  (dim={dim})")
        print("=" * 60)

        if vectors is not None:
            embed_matrix, vocab_size = build_embedding_matrix(
                word_index, vectors, dim, mode_name,
            )
        else:
            vocab_size = min(MAX_VOCAB, len(word_index) + 1)
            embed_matrix = None

        results[mode_name] = {
            "embedding_dim": dim,
            "vocab_size": vocab_size,
            "models": {},
        }

        for arch_name, builder, train_X, test_X, trip_X, maxlen in [
            ("original", build_original_cnn, train50, test50, trip50,
             MAX_SEQUENCE_LENGTH_ORIG),
            ("improved", build_improved_cnn, train100, test100, trip100,
             MAX_SEQUENCE_LENGTH_IMPROVED),
        ]:
            print(f"\n--- {arch_name.upper()} CNN with {mode_name} embeddings ---")
            tf.keras.backend.clear_session()
            np.random.seed(42); tf.random.set_seed(42)
            model = builder(vocab_size, dim, embed_matrix)
            es = EarlyStopping(monitor="val_loss", patience=5,
                               restore_best_weights=True)
            model.fit(
                train_X, y_train,
                epochs=EPOCHS, batch_size=BATCH_SIZE,
                validation_split=0.1, callbacks=[es], verbose=2,
            )
            imdb_loss, imdb_acc = model.evaluate(test_X, y_test, verbose=0)
            trip_loss, trip_acc = model.evaluate(trip_X, y_trip, verbose=0)
            print(f"  IMDB        acc={imdb_acc:.4f}  loss={imdb_loss:.4f}")
            print(f"  TripAdvisor acc={trip_acc:.4f}  loss={trip_loss:.4f}")

            model_path = os.path.join(
                OUT_DIR, f"{arch_name}_cnn_{mode_name}.keras",
            )
            model.save(model_path)

            results[mode_name]["models"][arch_name] = {
                "imdb_test":   {"loss": float(imdb_loss),
                                "accuracy": float(imdb_acc),
                                "details": detailed_eval(model, test_X,
                                                         y_test_int)},
                "tripadvisor": {"loss": float(trip_loss),
                                "accuracy": float(trip_acc),
                                "details": detailed_eval(model, trip_X,
                                                         y_trip_int)},
                "model_file": os.path.basename(model_path),
                "maxlen": maxlen,
            }

    with open(os.path.join(OUT_DIR, "tokenizer.pkl"), "wb") as f:
        pickle.dump(tokenizer, f)

    config = {
        "MAX_SEQUENCE_LENGTH_ORIG": MAX_SEQUENCE_LENGTH_ORIG,
        "MAX_SEQUENCE_LENGTH_IMPROVED": MAX_SEQUENCE_LENGTH_IMPROVED,
        "MAX_VOCAB": MAX_VOCAB,
        "available_modes": [m[0] for m in modes],
        "embedding_dims": {m[0]: m[1] for m in modes},
    }
    with open(os.path.join(OUT_DIR, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    with open(os.path.join(OUT_DIR, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    print(f"{'Embedding':<12}{'Architecture':<14}{'IMDB acc':<11}"
          f"{'IMDB loss':<11}{'Trip acc':<11}{'Trip loss':<11}")
    print("-" * 80)
    for mode_name, _, _ in modes:
        for arch_name in ("original", "improved"):
            m = results[mode_name]["models"][arch_name]
            print(f"{mode_name:<12}{arch_name:<14}"
                  f"{m['imdb_test']['accuracy']:<11.4f}"
                  f"{m['imdb_test']['loss']:<11.4f}"
                  f"{m['tripadvisor']['accuracy']:<11.4f}"
                  f"{m['tripadvisor']['loss']:<11.4f}")
    print("=" * 80)
    print("Done.")

if __name__ == "__main__":
    main()
