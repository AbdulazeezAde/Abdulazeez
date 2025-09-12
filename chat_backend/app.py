# app.py - uses Gemini embeddings for query encoding (requires FAISS index built with Gemini embeddings)
import os
import pickle
import traceback
from flask import Flask, request, jsonify
from flask_cors import CORS

import numpy as np
import faiss

# Langchain helper to call Gemini embeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# For Gemini generation (keeps using the SDK)
import google.generativeai as genai

# Config (set these as environment variables on Render)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
# The embedding model you used to build the index. Must match the model used when building the FAISS index.
GEMINI_EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "models/embedding-001")

FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "faiss_index.index")
TEXTS_PKL_PATH = os.getenv("TEXTS_PKL_PATH", "texts.pkl")
TOP_K = int(os.getenv("TOP_K", "3"))

app = Flask(__name__)
CORS(app)

# === Load FAISS index and texts at startup (no heavy embedding calls) ===
INDEX = None
TEXTS = None
_index_dim = None
_load_error = None

def load_index_and_texts():
    global INDEX, TEXTS, _index_dim, _load_error
    try:
        if not os.path.exists(FAISS_INDEX_PATH) or not os.path.exists(TEXTS_PKL_PATH):
            _load_error = f"Index or texts not found at {FAISS_INDEX_PATH} / {TEXTS_PKL_PATH}"
            print(_load_error)
            return

        # Read FAISS index. If the index was created with normalized vectors (for cosine search via inner product),
        # ensure you built it that way and queries will be normalized too.
        INDEX = faiss.read_index(FAISS_INDEX_PATH)
        _index_dim = INDEX.d
        print(f"Loaded FAISS index with dimension {_index_dim} and {INDEX.ntotal} vectors.")

        with open(TEXTS_PKL_PATH, "rb") as f:
            TEXTS = pickle.load(f)
        print(f"Loaded {len(TEXTS)} texts.")

    except Exception as e:
        _load_error = f"Failed to load index/texts: {e}\n{traceback.format_exc()}"
        print(_load_error)

# Load at import/startup (fast; doesn't call embedding API)
load_index_and_texts()

# Configure Gemini for generation (only used for generate calls)
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

def normalize_vector(vec: np.ndarray):
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm

def embed_query_with_gemini(query: str):
    """Use LangChain's GoogleGenerativeAIEmbeddings to embed a single query using Gemini.
       Returns a numpy array vector or raises an exception."""
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY not set")

    try:
        emb_client = GoogleGenerativeAIEmbeddings(model=GEMINI_EMBEDDING_MODEL, google_api_key=GEMINI_API_KEY)
        # embed_documents accepts a list; for a single query we pass [query]
        vectors = emb_client.embed_documents([query])  # returns list[list[float]]
        if not vectors or not isinstance(vectors, (list, tuple)):
            raise RuntimeError(f"Unexpected embeddings response: {vectors}")
        vec = np.asarray(vectors[0], dtype="float32")
        return vec
    except Exception as e:
        # Bubble up — caller will detect rate-limit text and handle it
        raise

def retrieve_context(query: str, k: int = TOP_K):
    """Embed the query using Gemini, then search the FAISS index and return top-k texts joined."""
    if _load_error:
        return None, _load_error
    if INDEX is None or TEXTS is None:
        return None, "Index/texts not loaded on server."

    try:
        q_vec = embed_query_with_gemini(query)

        # Dimension check: ensure embedding and index dimensions match
        if q_vec.shape[0] != _index_dim:
            return None, (
                f"Embedding dimension mismatch: index dimension={_index_dim}, "
                f"query embedding dimension={q_vec.shape[0]}. "
                "This means the index was built with a different embedding model. Rebuild the index with Gemini embeddings."
            )

        # Normalize if index was built with normalized vectors (common for cosine via inner-product)
        q_vec = normalize_vector(q_vec).astype("float32")
        q_vec = np.expand_dims(q_vec, axis=0)

        D, I = INDEX.search(q_vec, k)
        idxs = I[0].tolist()
        results = []
        for idx in idxs:
            if idx < 0 or idx >= len(TEXTS):
                continue
            results.append(TEXTS[idx])
        return "\n---\n".join(results), None

    except Exception as e:
        s = str(e).lower()
        # surfacing rate-limit messages
        if "429" in s or "rate limit" in s or "quota" in s or "too many requests" in s:
            return None, "Gemini embedding rate limit / quota error. Please wait or increase quota."
        return None, f"Error embedding query with Gemini: {e}"

def is_rate_limit_exception(e: Exception) -> bool:
    s = str(e).lower()
    return ("429" in s) or ("rate limit" in s) or ("quota" in s) or ("too many requests" in s)

def ask_gemini_with_context(question: str, context: str):
    if not GEMINI_API_KEY:
        return "Gemini API key not set on server."

    try:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel("gemini-2.5-flash-lite")
        prompt = (
            "You are an assistant with access to details about me. Don't mention the resume. "
            "Answer the following question based ONLY on the provided context. "
            "If the answer is not present, say 'I could not find that information about Abdulazeez.'\n\n"
            f"Context:\n{context}\n\nQuestion: {question}\n\n"
            "If the context is long, summarize or return only the most relevant information."
        )
        response = model.generate_content(prompt)
        if hasattr(response, "text"):
            return response.text.strip()
        if hasattr(response, "candidates") and response.candidates:
            return getattr(response.candidates[0], "output", str(response.candidates[0])).strip()
        return str(response)
    except Exception as e:
        if is_rate_limit_exception(e):
            return ("Gemini rate limit or quota error encountered during generation. "
                    "Please wait or contact support / increase quota.")
        return f"Error calling Gemini for generation: {e}"

@app.route("/", methods=["GET"])
def home():
    return "Abdulazeez Chat API (FAISS-backed, Gemini embeddings for queries) is running."

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json() or {}
    question = data.get("message", "").strip()
    if not question:
        return jsonify({"error": "No message provided"}), 400

    context, err = retrieve_context(question, k=TOP_K)
    if context is None:
        # return error so client knows why retrieval failed
        return jsonify({"answer": err}), 503

    # Call Gemini for generation (one API call)
    answer = ask_gemini_with_context(question, context)
    return jsonify({"answer": answer})
