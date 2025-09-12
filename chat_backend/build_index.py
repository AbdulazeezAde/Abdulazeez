# build_index.py
import os
import pickle
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

# Config
RESUME_PATH = os.getenv("RESUME_PATH", "Resume.txt")
OUT_INDEX_PATH = os.getenv("OUT_INDEX_PATH", "faiss_index.index")
OUT_TEXTS_PATH = os.getenv("OUT_TEXTS_PATH", "texts.pkl")
MODEL_NAME = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "400"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "40"))

def load_text(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def chunk_text(text, chunk_size=400, overlap=40):
    # Simple character-based splitter similar to RecursiveCharacterTextSplitter behavior.
    if not text:
        return []
    chunks = []
    start = 0
    L = len(text)
    while start < L:
        end = min(start + chunk_size, L)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == L:
            break
        start = max(0, end - overlap)
    return chunks

def normalize_embeddings(vectors: np.ndarray):
    # Normalize rows to unit length for cosine search using inner product
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms

def main():
    print("Loading text...")
    text = load_text(RESUME_PATH)

    print("Chunking...")
    chunks = chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
    print(f"Created {len(chunks)} chunks.")

    print(f"Loading embedding model: {MODEL_NAME} ...")
    model = SentenceTransformer(MODEL_NAME)

    print("Encoding chunks (this may take a while)...")
    embs = model.encode(chunks, show_progress_bar=True, convert_to_numpy=True, batch_size=64)
    embs = normalize_embeddings(embs).astype("float32")

    d = embs.shape[1]
    print(f"Embedding dimension = {d}")

    print("Building FAISS index (IndexFlatIP for cosine similarity on normalized vectors) ...")
    index = faiss.IndexFlatIP(d)
    index.add(embs)
    print(f"Index has {index.ntotal} vectors.")

    print(f"Saving index to {OUT_INDEX_PATH} ...")
    faiss.write_index(index, OUT_INDEX_PATH)

    print(f"Saving texts to {OUT_TEXTS_PATH} ...")
    with open(OUT_TEXTS_PATH, "wb") as f:
        pickle.dump(chunks, f)

    print("Done. Copy faiss_index.index and texts.pkl to your Render service (or into your repo).")

if __name__ == "__main__":
    main()
