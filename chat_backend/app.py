import os
import json
import numpy as np
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from flask_cors import CORS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

# Load environment variables from .env
load_dotenv()

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
RESUME_PATH = os.getenv('RESUME_PATH', 'resume.txt')

app = Flask(__name__)
CORS(app)

class SimpleTfidfEmbeddings:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.chunks = []
        self.vectors = None
    
    def fit_transform(self, texts):
        self.chunks = texts
        self.vectors = self.vectorizer.fit_transform(texts)
        return self.vectors
    
    def similarity_search(self, query, k=3):
        if self.vectors is None:
            return []
        
        query_vector = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.vectors).flatten()
        
        # Get top k indices
        top_indices = similarities.argsort()[-k:][::-1]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 0:  # Only return if there's some similarity
                results.append({
                    'page_content': self.chunks[idx],
                    'score': similarities[idx]
                })
        
        return results

# Load and split resume into chunks
def load_resume_chunks():
    if os.path.exists(RESUME_PATH):
        try:
            with open(RESUME_PATH, 'r', encoding='utf-8') as f:
                text = f.read()
        except Exception as e:
            text = f"Error reading TXT: {e}"
    else:
        text = "Resume not found."
    
    # Simple chunking by sentences/paragraphs
    chunks = []
    paragraphs = text.split('\n\n')
    
    for para in paragraphs:
        if len(para.strip()) > 50:  # Only include substantial paragraphs
            # Split long paragraphs
            if len(para) > 400:
                sentences = para.split('. ')
                current_chunk = ""
                for sentence in sentences:
                    if len(current_chunk + sentence) < 400:
                        current_chunk += sentence + ". "
                    else:
                        if current_chunk.strip():
                            chunks.append(current_chunk.strip())
                        current_chunk = sentence + ". "
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
            else:
                chunks.append(para.strip())
    
    return chunks

# Initialize the embedding system
RESUME_CHUNKS = load_resume_chunks()
EMBEDDINGS = SimpleTfidfEmbeddings()

if RESUME_CHUNKS:
    EMBEDDINGS.fit_transform(RESUME_CHUNKS)

def ask_gemini(question):
    if not GEMINI_API_KEY:
        return "Gemini API key not set."
    
    if not RESUME_CHUNKS:
        return "Resume not loaded."
    
    try:
        # Get relevant chunks using TF-IDF similarity
        docs = EMBEDDINGS.similarity_search(question, k=3)
        
        if not docs:
            return "I could not find that information about Abdulazeez."
        
        rag_context = "\n---\n".join([d['page_content'] for d in docs])
        
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        prompt = (
            "You are an assistant with access to details about me. Don't mention the resume. "
            "Answer the following question based ONLY on the provided context. "
            "If the query or question is 'hello' or something general, say 'ask questions about Abdulazeez and what he does.'\n\n"
            "If the answer is not present, say 'Abdulazeez is an AI/ML Engineer and Researcher and a Data Scientist, what else would you like to know.'\n\n"
            f"Context:\n{rag_context}\n\nQuestion: {question}\n\n"
            "If the context is long, summarize or return only the most relevant information."
        )
        
        response = model.generate_content(prompt)
        return response.text.strip() if hasattr(response, 'text') else str(response)
        
    except Exception as e:
        return f"Error with Gemini API: {e}"

@app.route('/', methods=['GET'])
def home():
    return "Abdulazeez Chat API is running.", 200

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    question = data.get('message', '')
    
    if not question:
        return jsonify({'error': 'No message provided'}), 400
    
    answer = ask_gemini(question)
    return jsonify({'answer': answer})
"""
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    app.run(host='0.0.0.0', port=port, debug=False)
"""
