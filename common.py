from pathlib import Path

PROJECT_PATH = Path(".").absolute()

FAST_TEXT_MODEL_PATH = PROJECT_PATH / "data" / "embeddings" / "fasttext_embeddings.pkl"
SKIPGRAM_TEXT_MODEL_PATH = PROJECT_PATH / "data" / "embeddings" / "skipgram_embeddings.pkl"
