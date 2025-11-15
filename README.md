# ChatGPT Clone (Gemini) — Fixed for Render

This repository is a minimal Flask app that uses Google Gemini (google-generativeai).
Fixes included:
- `templates/index.html` placed in `templates/` so Flask can find it.
- `requirements.txt` includes `gunicorn`.
- Added `Procfile` for Render start command.
- `app.py` updated to listen on $PORT and handle missing GEMINI_API_KEY more gracefully.

## Setup locally

1. Create venv & install:
```
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

2. Create `.env` from `.env.example` and add your GEMINI_API_KEY, or set env var in hosting.

3. Run locally:
```
python app.py
```

## Deploy to Render

- Create a Web Service (not Static Site).
- Build command: `pip install -r requirements.txt`
- Start command (or leave blank if using Procfile): `gunicorn app:app --bind 0.0.0.0:$PORT`
- Add environment variable `GEMINI_API_KEY` in Render dashboard.

