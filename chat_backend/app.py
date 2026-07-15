import hashlib
import json
import os
import re
import time
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


BASE_DIR = Path(__file__).resolve().parent
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

RESUME_PATH = BASE_DIR / "Resume.txt"
KNOWLEDGE_PATH = BASE_DIR / "knowledge.json"
CONTACT_MESSAGES_PATH = BASE_DIR / "contact_messages.jsonl"

app = Flask(__name__)
CORS(app)


def normalize_text(value):
    return re.sub(r"\s+", " ", str(value or "")).strip()


def normalize_query(value):
    return normalize_text(value).lower()


def clean_contact_field(value, max_length):
    return normalize_text(value)[:max_length]


def append_contact_message(record):
    with open(CONTACT_MESSAGES_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=True) + "\n")


def load_knowledge():
    if not KNOWLEDGE_PATH.exists():
        return {}
    with open(KNOWLEDGE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


KNOWLEDGE = load_knowledge()


def read_resume_text():
    if not RESUME_PATH.exists():
        return ""
    with open(RESUME_PATH, "r", encoding="utf-8") as f:
        return f.read()


def split_resume_chunks(text):
    chunks = []
    current_heading = ""

    for block in [part.strip() for part in text.split("\n\n") if part.strip()]:
        if block.startswith("#"):
            current_heading = block.strip("# ").strip()
            continue

        if len(block) < 35:
            continue

        if len(block) > 650:
            sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", block) if s.strip()]
            current = []
            current_len = 0
            for sentence in sentences:
                if current_len + len(sentence) > 500 and current:
                    chunks.append(f"{current_heading}: {' '.join(current)}".strip(": "))
                    current = [sentence]
                    current_len = len(sentence)
                else:
                    current.append(sentence)
                    current_len += len(sentence)
            if current:
                chunks.append(f"{current_heading}: {' '.join(current)}".strip(": "))
        else:
            chunks.append(f"{current_heading}: {block}".strip(": "))

    return dedupe(chunks)


def dedupe(items):
    seen = set()
    unique = []
    for item in items:
        key = hashlib.md5(item.encode("utf-8")).hexdigest()
        if key not in seen:
            unique.append(item)
            seen.add(key)
    return unique


def flatten_knowledge_chunks(knowledge):
    chunks = []

    profile = knowledge.get("profile", {})
    if profile:
        chunks.append("Profile: " + " ".join(str(v) for v in profile.values() if v))

    contact = knowledge.get("contact", {})
    if contact:
        chunks.append("Contact: " + ", ".join(f"{k}: {v}" for k, v in contact.items()))

    for item in knowledge.get("experience", []):
        bullets = " ".join(item.get("highlights", []))
        chunks.append(f"Experience: {item.get('role')} at {item.get('company')} ({item.get('period')}). {bullets}")

    education = knowledge.get("education", {})
    if education:
        chunks.append("Education: " + " ".join(str(v) for v in education.values() if v))

    for group, skills in knowledge.get("skills", {}).items():
        chunks.append(f"Skills - {group}: {', '.join(skills)}")

    for project in knowledge.get("projects", []):
        sections = " ".join(project.get("details", []))
        chunks.append(
            f"Project: {project.get('name')}. Category: {project.get('category')}. "
            f"Summary: {project.get('summary')} Stack: {', '.join(project.get('stack', []))}. {sections}"
        )

    for section in ("community", "research", "links"):
        value = knowledge.get(section)
        if isinstance(value, list):
            chunks.append(f"{section.title()}: " + "; ".join(value))
        elif isinstance(value, dict):
            chunks.append(f"{section.title()}: " + ", ".join(f"{k}: {v}" for k, v in value.items()))

    return dedupe([normalize_text(chunk) for chunk in chunks if normalize_text(chunk)])


class LocalRetriever:
    def __init__(self, chunks):
        self.chunks = chunks
        self.cache = {}
        self.vectorizer = TfidfVectorizer(
            max_features=1200,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95,
            norm="l2",
        )
        self.vectors = None
        if chunks:
            self.vectors = self.vectorizer.fit_transform(chunks).astype(np.float32)

    def search(self, query, k=4, min_score=0.08):
        if self.vectors is None:
            return []

        cache_key = f"{hashlib.md5(query.encode('utf-8')).hexdigest()}:{k}:{min_score}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        query_vector = self.vectorizer.transform([query]).astype(np.float32)
        similarities = cosine_similarity(query_vector, self.vectors).flatten()
        top_indices = np.argsort(similarities)[::-1][:k]

        results = [
            {"content": self.chunks[idx], "score": float(similarities[idx])}
            for idx in top_indices
            if similarities[idx] >= min_score
        ]

        self.cache[cache_key] = results
        if len(self.cache) > 120:
            for key in list(self.cache.keys())[:30]:
                del self.cache[key]
        return results


RESUME_TEXT = read_resume_text()
RESUME_CHUNKS = split_resume_chunks(RESUME_TEXT)
KNOWLEDGE_CHUNKS = flatten_knowledge_chunks(KNOWLEDGE)
ALL_CHUNKS = dedupe(KNOWLEDGE_CHUNKS + RESUME_CHUNKS)
RETRIEVER = LocalRetriever(ALL_CHUNKS)
RESPONSE_CACHE = {}

print(f"Loaded {len(KNOWLEDGE_CHUNKS)} knowledge chunks and {len(RESUME_CHUNKS)} resume chunks")


def is_greeting(query):
    return bool(re.fullmatch(r"(hi|hello|hey|good morning|good afternoon|good evening|yo|sup)[!. ]*", query))


def contains_any(query, words):
    return any(word in query for word in words)


def find_project(query):
    projects = KNOWLEDGE.get("projects", [])
    for project in projects:
        aliases = [project.get("name", "")] + project.get("aliases", [])
        if any(normalize_query(alias) and normalize_query(alias) in query for alias in aliases):
            return project
    return None


def format_list(items):
    return "\n".join(f"- {item}" for item in items if item)


def format_project(project):
    lines = [
        f"{project['name']} is a {project.get('category', 'portfolio')} project.",
        project.get("summary", ""),
    ]
    if project.get("stack"):
        lines.append("Tools/stack: " + ", ".join(project["stack"]) + ".")
    if project.get("details"):
        lines.append("Key details:\n" + format_list(project["details"]))
    if project.get("url"):
        lines.append(f"Project link: {project['url']}")
    return "\n\n".join(line for line in lines if line)


def answer_from_intent(question):
    query = normalize_query(question)

    if is_greeting(query):
        return "Hello, what would you like to know about Abdulazeez?"

    if contains_any(query, ["contact", "email", "phone", "linkedin", "github", "reach", "hire"]):
        contact = KNOWLEDGE.get("contact", {})
        if not contact:
            return None
        return (
            "You can contact Abdulazeez through:\n"
            + format_list(
                [
                    f"Email: {contact.get('email')}",
                    f"Phone: {contact.get('phone')}",
                    f"LinkedIn: {contact.get('linkedin')}",
                    f"GitHub: {contact.get('github')}",
                ]
            )
        )

    project = find_project(query)
    if project:
        return format_project(project)

    if contains_any(query, ["project", "portfolio", "built", "work on", "worked on"]):
        projects = KNOWLEDGE.get("projects", [])
        return "Selected projects:\n" + format_list(
            [f"{p['name']}: {p.get('summary', '')}" for p in projects]
        )

    if contains_any(query, ["skill", "tools", "stack", "technology", "technologies"]):
        skills = KNOWLEDGE.get("skills", {})
        if not skills:
            return None
        return "Abdulazeez works with:\n" + "\n".join(
            f"- {group.title()}: {', '.join(values)}" for group, values in skills.items()
        )

    if contains_any(query, ["experience", "job", "work history", "employment", "role"]):
        experience = KNOWLEDGE.get("experience", [])
        return "Work experience:\n" + format_list(
            [
                f"{item.get('role')} at {item.get('company')} ({item.get('period')}): "
                + " ".join(item.get("highlights", [])[:2])
                for item in experience
            ]
        )

    if contains_any(query, ["education", "degree", "school", "university", "cgpa", "study", "studied"]):
        education = KNOWLEDGE.get("education", {})
        if not education:
            return None
        return (
            f"Abdulazeez studied {education.get('degree')} at {education.get('university')}. "
            f"Classification: {education.get('classification')}. "
            f"Additional courses include {', '.join(education.get('courses', []))}."
        )

    if contains_any(query, ["community", "volunteer", "leadership", "group"]):
        community = KNOWLEDGE.get("community", [])
        if community:
            return "Community involvement:\n" + format_list(community)

    if contains_any(query, ["research", "publication", "african nlp", "yoruba", "language"]):
        research = KNOWLEDGE.get("research", [])
        if research:
            return "Research interests and work:\n" + format_list(research)

    return None


def compose_retrieval_answer(question):
    docs = RETRIEVER.search(question, k=4)
    if not docs:
        return "I could not find that information about Abdulazeez. What else would you like to know?"

    snippets = []
    for doc in docs[:3]:
        content = doc["content"]
        if len(content) > 520:
            content = content[:517].rstrip() + "..."
        snippets.append(content)

    return "Here is what I found about Abdulazeez:\n" + format_list(snippets)


def answer_question(question):
    question = normalize_text(question)
    if not question:
        return "Please ask a question about Abdulazeez."

    cache_key = hashlib.md5(question.lower().encode("utf-8")).hexdigest()
    if cache_key in RESPONSE_CACHE:
        return RESPONSE_CACHE[cache_key]

    start = time.time()
    answer = answer_from_intent(question) or compose_retrieval_answer(question)
    RESPONSE_CACHE[cache_key] = answer

    if len(RESPONSE_CACHE) > 80:
        for key in list(RESPONSE_CACHE.keys())[:20]:
            del RESPONSE_CACHE[key]

    print(f"Answered locally in {time.time() - start:.3f}s")
    return answer


@app.route("/", methods=["GET"])
def home():
    return f"Abdulazeez Chat API is running. Loaded {len(ALL_CHUNKS)} local knowledge chunks.", 200


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    question = data.get("message", "")

    if not normalize_text(question):
        return jsonify({"error": "No message provided"}), 400

    return jsonify({"answer": answer_question(question)})


@app.route("/contact", methods=["POST"])
def contact():
    payload = request.get_json(silent=True) if request.is_json else request.form

    name = clean_contact_field(payload.get("name", ""), 120)
    email = clean_contact_field(payload.get("email", ""), 180)
    subject = clean_contact_field(payload.get("subject", ""), 180)
    message = clean_contact_field(payload.get("message", ""), 4000)

    if not all([name, email, subject, message]):
        return "Please complete all required fields.", 400

    if not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", email):
        return "Please enter a valid email address.", 400

    record = {
        "received_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "name": name,
        "email": email,
        "subject": subject,
        "message": message,
    }
    append_contact_message(record)
    print(f"Contact form submission: {json.dumps(record, ensure_ascii=True)}")

    return "OK", 200


@app.route("/health", methods=["GET"])
def health():
    return jsonify(
        {
            "status": "healthy",
            "mode": "local-retrieval",
            "knowledge_chunks": len(KNOWLEDGE_CHUNKS),
            "resume_chunks": len(RESUME_CHUNKS),
            "total_chunks": len(ALL_CHUNKS),
            "retrieval_cache_size": len(RETRIEVER.cache),
            "response_cache_size": len(RESPONSE_CACHE),
        }
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port, debug=False)
