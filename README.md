# Disaster Tweet Verifier

A lightweight, public REST API + web UI that predicts whether a short text describes a real disaster.

- **Live demo:** `https://disaster-tweet-verifier.onrender.com` (replace with your deployed Render URL)
- **Git repo:** `https://github.com/USERNAME/disaster-tweet-verifier`

## What it does

Built for the *Natural Language Processing with Disaster Tweets*. The model is a `TF-IDF + soft-voting ensemble` trained on ~7,600 hand-labelled tweets, reaching **~0.79 validation F1**. It exposes two endpoints:

- `GET /health` → `{ "status": "ok" }`
- `POST /predict` → `{ "label": 0|1, "score": 0.0-1.0 }`
  - Request body: `{ "text": "...", "keyword": "..." }` (`keyword` is optional).
  - `label` uses an F1-optimal decision threshold tuned on validation (not a fixed 0.5).
  - `score` is the model's disaster probability for the input.

The root path (`/`) serves a single-page dark-themed UI where anyone can paste text, click **Verify**, and see the predicted label plus a confidence meter.

## Architecture

- **Features**: A scikit-learn `FeatureUnion` of word TF-IDF (1-2 grams, English stop words, sublinear TF) **and** character TF-IDF (`char_wb`, 2-5 grams). The `keyword` column is combined with the tweet text, and hashtag/mention words are preserved during cleaning (e.g. `#earthquake` → `earthquake`).
- **Classifier**: A soft-voting ensemble of `LogisticRegression` (balanced), `ComplementNB`, and a calibrated `LinearSVC`. A decision threshold is tuned on validation to maximize F1 and saved alongside the pipeline.
- **Model artifact**: `model.pkl` stores a dict `{ "pipeline", "threshold" }`. The full model is small (a few MB) and infers in milliseconds on a single CPU.
- **API**: FastAPI on Python 3.11. It validates JSON, handles CORS, and serves static files from the same process.
- **UI**: Vanilla HTML/CSS/JS with a dark theme, live character counter, confidence meter, and loading/error states.
- **Build**: The model is trained from the bundled `train.csv` during the build step, so no model file needs to be committed (`model.pkl` is git-ignored).
- **Deployment**: Render free web service (Python runtime, defined in `render.yaml`). A `Dockerfile` is also included if you prefer a container-based deploy.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python train.py
uvicorn app:app --reload
```

Then open `http://localhost:8000` and try the UI, or test the API:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Forest fire near La Ronge Sask. Canada", "keyword": "fire"}'
```

The dataset (`train.csv`) is bundled in the repo, so `train.py` reads it locally with no network access. If it is missing, `train.py` falls back to downloading it.

## Deploy to Render (free tier)

This repo includes a `render.yaml` Blueprint, so deployment is one click after pushing to GitHub.

1. Push this project to a GitHub repository.
2. Go to the [Render Dashboard](https://dashboard.render.com) → **New** → **Blueprint**.
3. Connect your repo. Render reads `render.yaml` and provisions a free web service.
4. On first deploy, Render runs the build command:
   ```
   pip install -r requirements.txt && python train.py
   ```
   `train.py` uses the bundled `train.csv`, trains the model, and writes `model.pkl`. Then it starts the API with:
   ```
   uvicorn app:app --host 0.0.0.0 --port $PORT
   ```
5. Once live, your service is available at `https://<service-name>.onrender.com`.

**Note:** Free Render services spin down after ~15 minutes of inactivity, so the first request after idle may take ~30-60 s to cold start. The `/health` endpoint is used as the health check.

## Design trade-offs

I chose a classical TF-IDF ensemble over a transformer because the task values a working, low-latency public API over cutting-edge F1. The model is tiny, fast to train, easy to audit, and fits comfortably within Render's free 512 MB tier with sub-second inference. Word + character n-grams handle noisy tweet text (typos, hashtags), the `keyword` column adds a strong signal, and tuning the decision threshold squeezes out extra F1 — reaching **~0.79** without the cost of a heavy neural model. The UI is intentionally minimal and self-hosted in the same process to keep deployment and CORS simple.
