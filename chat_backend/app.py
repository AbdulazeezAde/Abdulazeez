# app.py
import os
import pickle
import time
from flask import Flask, request, jsonify
from flask_cors import CORS

# For local query embedding
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

# For Gemini generation
import google.generativeai as genai

# Config (set these as environment variables on Render)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "faiss_index.index")
TEXTS_PKL_PATH = os.getenv("TEXTS_PKL_PATH", "texts.pkl")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
TOP_K = int(os.getenv("TOP_K", "3"))

app = Flask(__name__)
CORS(app)

# === Load FAISS index and texts at startup ===
INDEX = None
TEXTS = []
EMBEDDING_MODEL_INSTANCE = None

def load_index_and_texts():
    global INDEX, TEXTS, EMBEDDING_MODEL_INSTANCE
    if not os.path.exists(FAISS_INDEX_PATH) or not os.path.exists(TEXTS_PKL_PATH):
        print("FAISS index or texts not found. INDEX will be None.")
        INDEX = None
        TEXTS = []
        return

    print("Loading FAISS index...")
    INDEX = faiss.read_index(FAISS_INDEX_PATH)
    print("Loading texts...")
    with open(TEXTS_PKL_PATH, "rb") as f:
        TEXTS = pickle.load(f)

    print(f"Loaded index with {INDEX.ntotal} vectors and {len(TEXTS)} texts.")

    # Load local model for query encoding
    print(f"Loading SentenceTransformer model: {EMBEDDING_MODEL} ...")
    EMBEDDING_MODEL_INSTANCE = SentenceTransformer(EMBEDDING_MODEL)
    print("Model loaded.")

load_index_and_texts()

# configure Gemini for generation (only used for generate calls)
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

def normalize(v: np.ndarray):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm

def retrieve_context(query: str, k: int = TOP_K):
    """Encode query with local model, search FAISS and return top-k texts joined."""
    if INDEX is None or EMBEDDING_MODEL_INSTANCE is None:
        return None

    q_emb = EMBEDDING_MODEL_INSTANCE.encode([query], convert_to_numpy=True)[0].astype("float32")
    q_emb = normalize(q_emb)
    q_emb = np.expand_dims(q_emb, axis=0)

    # search (IndexFlatIP expects inner-product on normalized vectors for cosine)
    D, I = INDEX.search(q_emb, k)
    idxs = I[0].tolist()
    results = []
    for idx in idxs:
        if idx < 0 or idx >= len(TEXTS):
            continue
        results.append(TEXTS[idx])
    return "\n---\n".join(results)

def is_rate_limit_exception(e: Exception) -> bool:
    s = str(e).lower()
    return ("429" in s) or ("rate limit" in s) or ("quota" in s) or ("too many requests" in s)

def ask_gemini_with_context(question: str, context: str):
    if not GEMINI_API_KEY:
        return "Gemini API key not set on server."

    try:
        # try to generate content using Gemini SDK (same as earlier)
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
        # defensive handling of response shapes
        if hasattr(response, "text"):
            return response.text.strip()
        if hasattr(response, "candidates") and response.candidates:
            return getattr(response.candidates[0], "output", str(response.candidates[0])).strip()
        return str(response)
    except Exception as e:
        if is_rate_limit_exception(e):
            return ("Gemini rate limit or quota error encountered. "
                    "Please wait or contact support / increase quota.")
        return f"Error calling Gemini: {e}"

@app.route("/", methods=["GET"])
def home():
    return "Abdulazeez Chat API (FAISS-backed) is running."

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json() or {}
    question = data.get("message", "").strip()
    if not question:
        return jsonify({"error": "No message provided"}), 400

    # Retrieve context locally
    context = retrieve_context(question, k=TOP_K)
    if context is None:
        # If index missing, return helpful error so client can react
        return jsonify({"answer": ("Vector store not available on server. "
                                   "Precompute embeddings and upload faiss_index.index + texts.pkl.")}), 503

    # Call Gemini for generation (only one API call per chat)
    answer = ask_gemini_with_context(question, context)
    return jsonify({"answer": answer})
"""
if __name__ == "__main__":
    # For local testing
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5001)), debug=True)
"""
