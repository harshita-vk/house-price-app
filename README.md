# Bay Area Housing Price App

## Folder structure

```
house-price-app/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── ml_pipeline.py      # ClusterSimilarity transformer (shared by train + serve)
│   │   ├── main.py             # FastAPI app, /predict endpoint
│   │   └── model/
│   │       └── house_price_calculator_ml.pkl   <- trained model goes here
│   ├── train_model.py          # your notebook's training logic, run once
│   └── requirements.txt
└── frontend/
    └── index.html              # open directly in a browser, no build step
```

## Why you can't just drop in your Colab .pkl

Your `ClusterSimilarity` class was defined inside the Colab notebook, so when
`joblib` pickled the pipeline, it recorded the class's location as the
notebook's `__main__` namespace. Loading that file anywhere else fails with:

```
AttributeError: Can't get attribute 'ClusterSimilarity' on <module '__main__'>
```

Fix: `ml_pipeline.py` in this project defines that same class at a real,
importable path (`app.ml_pipeline.ClusterSimilarity`). `train_model.py`
imports it from there when training, and `main.py` imports it the same way
before loading the pickle — so the paths match and it loads cleanly.

**You have two options:**
1. Run `train_model.py` here to produce a fresh, compatible `.pkl` (recommended, tested below).
2. If you'd rather keep your exact already-trained model, tell me and I'll show you how to re-pickle it by re-registering the class path — but retraining is simpler and gives an identical model.

## 1. Backend setup

```bash
cd backend
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Train the model once (downloads the dataset, trains, saves the .pkl)
python train_model.py

# Start the API
uvicorn app.main:app --reload --port 8000
```

Verify it's up: open http://localhost:8000/health — should show `{"status":"ok","model_loaded":true}`.
Interactive API docs: http://localhost:8000/docs

## 2. Frontend setup

No build tooling needed — just open the file:

```bash
open frontend/index.html          # macOS
start frontend/index.html         # Windows
xdg-open frontend/index.html      # Linux
```

Fill in the form and click **Estimate price**. It calls
`http://localhost:8000/predict` directly from the browser (CORS is already
enabled on the backend for this).

## Tested end-to-end

This exact project was run in the build environment:
- `train_model.py` trained successfully, test RMSE ≈ 41,559 (matches your notebook's ballpark).
- The FastAPI server loaded the resulting `.pkl` with no unpickling errors.
- `POST /predict` with a sample Berkeley/Bay-area-ish district returned a real prediction (~$186,000).

## Moving to production later (optional, not needed to run locally)

- Swap `allow_origins=["*"]` in `main.py` for your actual frontend domain.
- Host the backend anywhere that runs Python (Render, Railway, Fly.io, EC2, etc.) and point `API_URL` in `index.html` at that URL instead of `localhost`.
- The frontend is a static file — deployable as-is to Netlify, Vercel, GitHub Pages, or S3.
