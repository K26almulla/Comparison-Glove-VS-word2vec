# Setting up GloVe and Word2Vec embeddings

The app supports **3 embedding modes**:

1. **Scratch** — trained from data, no extra files needed (already works out of the box)
2. **GloVe** — needs `glove.6B.100d.txt` (~330 MB)
3. **Word2Vec** — needs `GoogleNews-vectors-negative300.bin.gz` (~1.6 GB)

You don't need all 3. The training script auto-detects which files are present
and trains accordingly. The app shows a comparison of whatever is available.

## Where to put the files

Place them **in the same folder as `train_models.py`** (i.e., your `sentiment_app/` folder):

```
sentiment_app/
├── train_models.py
├── app.py
├── glove.6B.100d.txt              ← put here
├── GoogleNews-vectors-negative300.bin.gz   ← put here
├── artifacts/
└── ...
```

## Option 1 — GloVe (recommended start, smaller file)

### Download
1. Go to https://nlp.stanford.edu/projects/glove/
2. Download **`glove.6B.zip`** (the link says "Wikipedia 2014 + Gigaword 5", ~822 MB zip)
3. Unzip it — inside you'll find `glove.6B.50d.txt`, `glove.6B.100d.txt`, `glove.6B.200d.txt`, `glove.6B.300d.txt`
4. Move **`glove.6B.100d.txt`** into your `sentiment_app/` folder
5. Delete the zip and the other dimension files if you don't need them

### Or from command line
```bash
cd sentiment_app
curl -O https://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip glove.6B.100d.txt
rm glove.6B.zip
# Optional: also get the 50d, 200d, 300d if you want more variants
```

## Option 2 — Word2Vec (matches the lab exactly, larger file)

### Download
This is the file the lab points to: `GoogleNews-vectors-negative300.bin.gz`

The official link is on https://code.google.com/archive/p/word2vec/ but the
direct Google Drive link is unreliable. Easiest mirror is via gensim:

```bash
# Activate your venv first
source venv/bin/activate
pip install gensim

# This downloads ~1.6 GB to ~/gensim-data/
python3 -c "import gensim.downloader as api; api.load('word2vec-google-news-300')"

# Then copy it into your project folder, renamed to match what the script expects
cp ~/gensim-data/word2vec-google-news-300/word2vec-google-news-300.gz GoogleNews-vectors-negative300.bin.gz
```

Or download manually from one of the mirrors:
- https://huggingface.co/fse/word2vec-google-news-300 (need to download `word2vec-google-news-300.model` — but format is different, prefer the gensim method above)

## After downloading — re-train

```bash
# Make sure your venv is active
source venv/bin/activate

# Install gensim if you haven't (needed for Word2Vec)
pip install gensim

# Re-run training — auto-detects which files exist
python3 train_models.py
```

The script prints which embeddings it found:
```
Loading GloVe from /path/to/glove.6B.100d.txt...
  Loaded 400000 GloVe vectors, dim=100
Loading Word2Vec from /path/to/GoogleNews-vectors-negative300.bin.gz...
  Loaded 3000000 Word2Vec vectors, dim=300

Training with embedding modes: ['scratch', 'glove', 'word2vec']
```

Each mode trains 2 models (original + improved), so:
- 1 mode = 2 models trained (~30 sec)
- 2 modes = 4 models trained (~1 min)
- 3 modes = 6 models trained (~1.5 min)

## After training — relaunch the app

```bash
streamlit run app.py
```

Now the sidebar will show all available embedding modes, and the
**"Embedding comparison"** tab will show side-by-side metrics and a bar chart.

## What if something goes wrong?

**"GloVe file not found"** — double-check the filename is exactly
`glove.6B.100d.txt` and it's in the same folder as `train_models.py`.

**"Word2Vec file not found"** — same; the script looks for the exact name
`GoogleNews-vectors-negative300.bin.gz`. If your file has a different name,
either rename it or edit the `WORD2VEC_PATH` line near the top of
`train_models.py`.

**Training is very slow** — Word2Vec loading takes 30–60 seconds because the
file is huge. After it loads, the actual training is the same speed as before.
