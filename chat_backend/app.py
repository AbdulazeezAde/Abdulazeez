import os
import json
import numpy as np
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from flask_cors import CORS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings.base import Embeddings
from sentence_transformers import SentenceTransformer

# Load environment variables from .env
load_dotenv()

# Multiple API keys - add your keys here
GEMINI_API_KEY =  os.getenv('GEMINI_API_KEY')

RESUME_PATH = os.getenv('RESUME_PATH', 'resume.txt')
current_key_index = 0

app = Flask(__name__)
CORS(app)

# Custom embedding class for sentence-transformers
class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        # This model is only ~80MB and very fast
        self.model = SentenceTransformer(model_name)
    
    def embed_documents(self, texts):
        return self.model.encode(texts).tolist()
    
    def embed_query(self, text):
        return self.model.encode([text])[0].tolist()

# Load and split resume into chunks for RAG
def load_resume_chunks():
    if os.path.exists(RESUME_PATH):
        try:
            with open(RESUME_PATH, 'r', encoding='utf-8') as f:
                text = f.read()
        except Exception as e:
            text = f"Error reading TXT: {e}"
    else:
        text = "Resume not found."
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=40)
    return splitter.split_text(text)

RESUME_CHUNKS = load_resume_chunks()

# Build vector store for RAG with local embeddings
def build_vectorstore(chunks):
    try:
        embeddings = SentenceTransformerEmbeddings()
        return FAISS.from_texts(chunks, embeddings)
    except Exception as e:
        print(f"Error building vectorstore: {e}")
        return None

VECTORSTORE = build_vectorstore(RESUME_CHUNKS) if RESUME_CHUNKS else None

# RAG with LangChain and Gemini
def ask_gemini(question):
    if not GEMINI_API_KEY:
        return "Gemini API key not set."
    
    if not VECTORSTORE:
        return "Vector store not initialized."
    
    try:
        # Retrieve top 3 relevant chunks
        docs = VECTORSTORE.similarity_search(question, k=3)
        rag_context = "\n---\n".join([d.page_content for d in docs])
        
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        prompt = (
            "You are an assistant with access to details about me. Don't mention the resume. "
            "Answer the following question based ONLY on the provided context. "
            "If the answer is not present, say 'I could not find that information about Abdulazeez.'\n\n"
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
    app.run(debug=True, port=5001)
"""
