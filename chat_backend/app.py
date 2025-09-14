import os
import pickle
import asyncio
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from flask_cors import CORS
import faiss
import numpy as np
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
import google.generativeai as genai

# Load environment variables
load_dotenv()

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
FAISS_INDEX_PATH = os.getenv('FAISS_INDEX_PATH', 'faiss_index.index')
TEXTS_PATH = os.getenv('TEXTS_PATH', 'texts.pkl')

app = Flask(__name__)
CORS(app)

# Global variables for fast access
faiss_index = None
resume_chunks = None
executor = ThreadPoolExecutor(max_workers=2)

# Initialize embedding model for queries only
try:
    from sentence_transformers import SentenceTransformer
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
except Exception as e:
    print(f"Error loading embedding model: {e}")
    embedding_model = None

def load_prebuilt_index():
    """Load pre-built FAISS index and texts"""
    global faiss_index, resume_chunks
    
    try:
        if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(TEXTS_PATH):
            # Load FAISS index
            faiss_index = faiss.read_index(FAISS_INDEX_PATH)
            
            # Load texts
            with open(TEXTS_PATH, 'rb') as f:
                resume_chunks = pickle.load(f)
            
            print(f"Loaded {len(resume_chunks)} chunks from pre-built index")
            return True
        else:
            print("Pre-built index files not found")
            return False
    except Exception as e:
        print(f"Error loading pre-built index: {e}")
        return False

# Load on startup
index_loaded = load_prebuilt_index()

@lru_cache(maxsize=100)
def cached_similarity_search(query_hash, k=3):
    """Cached similarity search to avoid repeated queries"""
    if not index_loaded or not embedding_model:
        return []
    
    try:
        # Convert hash back to query for embedding (in real use, you'd store embeddings)
        # This is a simplified version - in practice, you'd cache the embeddings
        query_embedding = embedding_model.encode([query_hash])
        query_embedding = np.array(query_embedding, dtype=np.float32)
        
        # Search FAISS index
        scores, indices = faiss_index.search(query_embedding, k)
        
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < len(resume_chunks) and score > 0.3:  # Similarity threshold
                results.append({
                    'page_content': resume_chunks[idx],
                    'score': float(score)
                })
        
        return results
    except Exception as e:
        print(f"Error in similarity search: {e}")
        return []

def fast_similarity_search(query, k=3):
    """Fast similarity search using pre-loaded FAISS index"""
    if not index_loaded or not embedding_model:
        return []
    
    try:
        # Embed query
        query_embedding = embedding_model.encode([query])
        query_embedding = np.array(query_embedding, dtype=np.float32)
        
        # Search FAISS index
        scores, indices = faiss_index.search(query_embedding, k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(resume_chunks) and score > 0.3:
                results.append({
                    'page_content': resume_chunks[idx],
                    'score': float(score)
                })
        
        return results
    except Exception as e:
        print(f"Error in similarity search: {e}")
        return []

@lru_cache(maxsize=50)
def cached_gemini_call(context_hash, question):
    """Cache Gemini API calls for repeated questions"""
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        prompt = (
            "You are an assistant with access to details about me. Don't mention the resume. "
            "Answer the following question based ONLY on the provided context. "
            "If the answer is not present, say 'I could not find that information about Abdulazeez.'\n\n"
            f"Context:\n{context_hash}\n\nQuestion: {question}\n\n"
            "If the context is long, summarize or return only the most relevant information."
        )
        
        response = model.generate_content(prompt)
        return response.text.strip() if hasattr(response, 'text') else str(response)
    except Exception as e:
        return f"Error with Gemini API: {e}"

def ask_gemini_async(question):
    """Async wrapper for Gemini API call"""
    if not GEMINI_API_KEY:
        return "Gemini API key not set."
    
    if not index_loaded:
        return "FAISS index not loaded."
    
    # Get relevant chunks using pre-loaded FAISS
    docs = fast_similarity_search(question, k=3)
    
    if not docs:
        return "I could not find that information about Abdulazeez."
    
    # Prepare context
    rag_context = "\n---\n".join([d['page_content'] for d in docs])
    
    # Use caching for repeated queries
    context_hash = hash(rag_context)
    return cached_gemini_call(str(context_hash), question)

@app.route('/', methods=['GET'])
def home():
    status = "loaded" if index_loaded else "not loaded"
    chunks_count = len(resume_chunks) if resume_chunks else 0
    return f"Abdulazeez Chat API is running. FAISS index: {status} ({chunks_count} chunks)", 200

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    question = data.get('message', '')
    
    if not question:
        return jsonify({'error': 'No message provided'}), 400
    
    # Use thread executor for non-blocking operation
    future = executor.submit(ask_gemini_async, question)
    answer = future.result(timeout=30)  # 30s timeout
    
    return jsonify({'answer': answer})

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'faiss_loaded': index_loaded,
        'chunks_count': len(resume_chunks) if resume_chunks else 0,
        'embedding_model_loaded': embedding_model is not None
    })
"""
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    app.run(host='0.0.0.0', port=port, debug=False)
"""
