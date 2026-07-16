# Chat Backend

## Setup
1. Keep `Resume.txt` and `knowledge.json` in this folder.
2. Create and activate a virtual environment:
   ```powershell
   python -m venv venv
   .\venv\Scripts\activate
   pip install -r production-requirements.txt
   ```
3. Run the server:
   ```powershell
   python app.py
   ```

## API
- POST `/chat` with JSON `{ "message": "your question" }`
- Returns `{ "answer": "..." }`
- POST `/contact` with form fields or JSON containing `name`, `email`, `subject`, and `message`
- Returns `OK` for successful contact submissions. This route is a fallback; the live portfolio contact form currently posts directly to Formspree.

## Note
- This backend does not require an LLM API key. It uses local structured knowledge plus TF-IDF retrieval, which is suitable for free Render hosting.
- Update `knowledge.json` when portfolio details change.
- The live portfolio contact form uses Formspree for email delivery. The backend `/contact` route only logs/appends submissions if you point a form to it again later.
