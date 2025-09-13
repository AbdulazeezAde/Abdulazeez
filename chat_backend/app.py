# app.py
import os
import pickle
import requests
import numpy as np
import faiss
from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai

# === Config (set these on Render) ===
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
HF_API_KEY = os.getenv("HF_API_KEY", "")
HF_MODEL = os.getenv("HF_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "faiss_index.index")
TEXTS_PKL_PATH = os.getenv("TEXTS_PKL_PATH", "texts.pkl")
TOP_K = int(os.getenv("TOP_K", "3"))

app = Flask(__name__)
CORS(app)

# === Globals ===
INDEX = None
TEXTS = []

# === Load FAISS index & texts at startup ===
def load_index_and_texts():
    global INDEX, TEXTS
    if not os.path.exists(FAISS_INDEX_PATH) or not os.path.exists(TEXTS_PKL_PATH):
        print("FAISS index or texts not found. INDEX will be None.")
        return

    print("Loading FAISS index...")
    INDEX = faiss.read_index(FAISS_INDEX_PATH)

    print("Loading texts...")
    with open(TEXTS_PKL_PATH, "rb") as f:
        TEXTS = pickle.load(f)

    print(f"Loaded index with {INDEX.ntotal} vectors and {len(TEXTS)} texts.")

load_index_and_texts()

# === Hugging Face Embeddings ===
def get_hf_embedding(text: str):
    """Call Hugging Face Inference API for embeddings"""
    if not HF_API_KEY:
        raise Exception("HF_API_KEY not set in environment.")

    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{HF_MODEL}"
    response = requests.post(url, headers=headers, json={"inputs": text}, timeout=30)

    if response.status_code != 200:
        raise Exception(f"HuggingFace API error: {response.status_code} {response.text}")

    embedding = response.json()
    # Flatten nested list
    if isinstance(embedding, list) and isinstance(embedding[0], list):
        return embedding[0]
    return embedding

def normalize(v: np.ndarray):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm

def retrieve_context(query: str, k: int = TOP_K):
    if INDEX is None:
        return None

    q_emb = np.array(get_hf_embedding(query), dtype="float32")
    q_emb = normalize(q_emb)
    q_emb = np.expand_dims(q_emb, axis=0)

    # Search FAISS
    D, I = INDEX.search(q_emb, k)
    idxs = I[0].tolist()
    results = [TEXTS[idx] for idx in idxs if 0 <= idx < len(TEXTS)]
    return "\n---\n".join(results)

# === Gemini Setup ===
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

def is_rate_limit_exception(e: Exception) -> bool:
    s = str(e).lower()
    return ("429" in s) or ("rate limit" in s) or ("quota" in s) or ("too many requests" in s)

def ask_gemini_with_context(question: str, context: str):
    if not GEMINI_API_KEY:
        return "Gemini API key not set on server."

    try:
        model = genai.GenerativeModel("gemini-2.5-flash-lite")
        prompt = (
            "You are an assistant with access to details about Abdulazeez. "
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
            return "Gemini rate limit or quota error. Please wait or increase quota."
        return f"Error calling Gemini: {e}"

# === Routes ===
@app.route("/", methods=["GET"])
def home():
    return "Abdulazeez Chat API (FAISS + HuggingFace embeddings + Gemini generation) is running."

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json() or {}
    question = data.get("message", "").strip()
    if not question:
        return jsonify({"error": "No message provided"}), 400

    # Retrieve context with HF embeddings
    try:
        context = retrieve_context(question, k=TOP_K)
    except Exception as e:
        return jsonify({"answer": f"Error retrieving context: {e}"}), 500

    if context is None:
        return jsonify({"answer": "Vector store not available. Upload faiss_index.index + texts.pkl."}), 503

    # Generate answer with Gemini
    answer = ask_gemini_with_context(question, context)
    return jsonify({"answer": answer})
