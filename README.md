
# Abdulazeez Ade's Portfolio

A personal portfolio and résumé website with a Flask-powered chat API. The website is built from static HTML, CSS, and JavaScript; the API answers portfolio questions using local knowledge files and TF-IDF retrieval.

## Project contents

- `index.html` — portfolio home page
- `blog.html`, `blog-post.html`, `portfolio-details.html`, `resume.html`, and `starter-page.html` — additional pages
- `assets/` — stylesheets, scripts, images, and vendor libraries
- `chat_backend/` — Flask chat API, knowledge data, and backend setup instructions
- `forms/` — contact form assets

## Run the website locally

Serve the repository root with any static file server. For example, with Python:

```powershell
python -m http.server 8000
```

Then open <http://localhost:8000> in a browser.

## Run the chat API locally

Python 3.11 is used by CI. From the repository root, create and activate a virtual environment and install the backend dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r chat_backend/production-requirements.txt
python chat_backend/app.py
```

The API starts on <http://localhost:5001>. On macOS or Linux, activate the environment with `source .venv/bin/activate`.

The backend uses `chat_backend/Resume.txt` and `chat_backend/knowledge.json`; update these files when portfolio information changes. No LLM API key is required.

### API endpoints

- `GET /health` — returns API health and retrieval statistics.
- `POST /chat` — accepts JSON such as `{"message": "What are your skills?"}` and returns an answer as JSON.
- `POST /contact` — accepts form data or JSON with `name`, `email`, `subject`, and `message`. This fallback endpoint appends submissions to a local file; the live portfolio form currently uses Formspree.

## CI/CD

The GitHub Actions workflow at `.github/workflows/ci.yml` runs on pushes, pull requests targeting `main`, and manual dispatch. It installs the backend dependencies, checks Python syntax and knowledge JSON, and smoke-tests the health and chat endpoints.

After CI succeeds, pushes to `main` deploy to Vercel if the following GitHub repository settings are configured:

- Variables: `VERCEL_ORG_ID` and `VERCEL_PROJECT_ID`
- Secret: `VERCEL_TOKEN`

Without these settings, CI still runs and the deployment job reports that deployment was skipped. The Vercel project configuration is in `chat_backend/vercel.json`.
