import os
import json
import faiss
import numpy as np
import requests
from flask import Flask, request, jsonify
from google import genai
from google.genai import types

# ------------------------------
# Config
# ------------------------------
HF_API_KEY = os.getenv("HF_API_KEY")  # set in Render dashboard
HF_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")  # set in Render dashboard
gemini_client = genai.Client(api_key=GEMINI_API_KEY)

# Flask app
app = Flask(__name__)

# ------------------------------
# Load FAISS index
# ------------------------------
INDEX_FILE = "faiss_index.bin"
TEXTS_FILE = "texts.json"

if os.path.exists(INDEX_FILE) and os.path.exists(TEXTS_FILE):
    index = faiss.read_index(INDEX_FILE)
    with open(TEXTS_FILE, "r") as f:
        texts = json.load(f)
    print(f"Loaded FAISS index with dimension {index.d} and {len(texts)} vectors.")
else:
    raise RuntimeError("FAISS index or texts.json not found. Run preprocessing first.")

# ------------------------------
# HuggingFace embedding function
# ------------------------------
def embed_texts_hf(texts_list):
    """Get embeddings from HuggingFace Inference API using feature-extraction pipeline."""
    try:
        headers = {"Authorization": f"Bearer {HF_API_KEY}"}
        url = f"https://api-inference.huggingface.co/models/{HF_MODEL}"

        responses = []
        for text in texts_list:
            resp = requests.post(url, headers=headers, json={"inputs": text})
            if resp.status_code == 200:
                embedding = np.array(resp.json()).mean(axis=0)  # average token embeddings
                responses.append(embedding)
            else:
                raise Exception(f"HuggingFace API error {resp.status_code}: {resp.text}")

        return np.array(responses).astype("float32")

    except Exception as e:
        print("Error retrieving context:", e)
        return np.zeros((len(texts_list), 384), dtype="float32")

# ------------------------------
# Context retrieval
# ------------------------------
def retrieve_context(query, k=3):
    query_vec = embed_texts_hf([query])
    D, I = index.search(query_vec, k)
    return [texts[i] for i in I[0] if i < len(texts)]

# ------------------------------
# Chat endpoint
# ------------------------------
@app.route("/chat", methods=["POST"])
def chat():
    try:
        user_input = request.json.get("message")
        if not user_input:
            return jsonify({"error": "No message provided"}), 400

        # Retrieve context
        context = retrieve_context(user_input)
        context_text = "\n".join(context)

        # Gemini for generation
        prompt = f"Use the context below to answer the question:\n\nContext:\n{context_text}\n\nQuestion: {user_input}\n\nAnswer:"
        response = gemini_client.models.generate_content(
            model="gemini-1.5-flash", contents=prompt
        )

        reply = response.text.strip()
        return jsonify({"response": reply, "context": context})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ------------------------------
# Render entrypoint (Gunicorn will run this)
# ------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)), debug=True)
