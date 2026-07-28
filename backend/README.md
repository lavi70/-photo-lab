# Product Photo Lab

Upload one product photo, pick a niche, get back 12 environment shots at 2000x2000px. Built as: FastAPI backend + a single static HTML/JS frontend (no build step, no npm required for the frontend).

## Setup

```
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn main:app --host 0.0.0.0 --port 8000
```
