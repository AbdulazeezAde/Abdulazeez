import os
import json
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from flask_cors import CORS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

# Load environment variables from .env
load_dotenv()
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
RESUME_PATH = os.getenv('RESUME_PATH', 'resume.txt')


app = Flask(__name__)
CORS(app)

# Load and split resume into chunks for RAG (now from JSON)

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

# Build vector store for RAG
def build_vectorstore(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GEMINI_API_KEY)
    return FAISS.from_texts(chunks, embeddings)

VECTORSTORE = build_vectorstore(RESUME_CHUNKS) if GEMINI_API_KEY and RESUME_CHUNKS else None


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
    return "Abdulazeez Chat API is running." , 200

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
