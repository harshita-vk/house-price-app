# Bay Area Housing Price Predictor

A machine learning model that predicts median house value for California
housing districts, trained on the classic StatLib California Housing
dataset, served through a FastAPI backend and a lightweight HTML/JS
frontend.

**Live demo:** https://deluxe-rabanadas-b41c76.netlify.app/

---

## ML Training

This is the core of the project — everything else (API, frontend) exists to
serve this model.

### Dataset

The [California Housing dataset](https://github.com/ageron/data) (20,640
rows), with one row per census district, containing:

| Feature | Description |
|---|---|
| `longitude`, `latitude` | Geographic location |
| `housing_median_age` | Median age of houses in the district |
| `total_rooms`, `total_bedrooms` | Total rooms/bedrooms across the district |
| `population`, `households` | Population and household counts |
| `median_income` | Median income (tens of thousands USD) |
| `ocean_proximity` | Categorical: `<1H OCEAN`, `INLAND`, `NEAR BAY`, `NEAR OCEAN`, `ISLAND` |
| `median_house_value` | **Target** — median house value for the district |

### Train/test split

A naive random split under-represents districts with unusual income
levels, since `median_income` is the strongest predictor of house value. To
fix this, the data is split using **stratified sampling** on an
income-bucketed category (5 income bins), so the train and test sets match
the overall income distribution:

```python
housing["income_cat"] = pd.cut(
    housing["median_income"],
    bins=[0., 1.5, 3.0, 4.5, 6.0, np.inf],
    labels=[1, 2, 3, 4, 5],
)
strat_train_set, strat_test_set = train_test_split(
    housing, test_size=0.2, stratify=housing["income_cat"], random_state=42
)
```

### Feature engineering

Built entirely inside a scikit-learn `Pipeline` / `ColumnTransformer`, so
preprocessing and the model are one serializable object — no separate
preprocessing script to keep in sync at inference time:

- **Ratio features** — `bedrooms_ratio` (bedrooms/rooms), `rooms_per_house`
  (rooms/households), `people_per_house` (population/households). Raw
  totals are noisy across districts of very different sizes; these ratios
  are far more predictive.
- **Log transforms** — applied to heavily right-skewed count features
  (`total_bedrooms`, `total_rooms`, `population`, `households`,
  `median_income`) before scaling, which stabilizes variance and helps
  linear/tree-based models alike.
- **Geographic similarity** — a custom `ClusterSimilarity` transformer
  (subclasses `BaseEstimator`, `TransformerMixin`) runs k-means on
  `[latitude, longitude]`, weighted by `median_house_value`, then scores
  each district by RBF-kernel similarity to each cluster center. This
  captures location effects (e.g. "near the Bay," "near a coastal cluster")
  far better than raw lat/long coordinates alone.
- **Categorical encoding** — `ocean_proximity` one-hot encoded, with
  `handle_unknown="ignore"` so unseen categories at inference time don't
  crash the pipeline.
- **Imputation + scaling** — median imputation for missing numeric values,
  `StandardScaler` throughout so no single feature dominates on scale.

### Model selection

Three model families were compared with the full feature pipeline attached,
using 10-fold cross-validated RMSE:

| Model | Cross-val RMSE (approx.) |
|---|---|
| Linear Regression | Highest error — underfits the non-linear location/price relationship |
| Decision Tree | Severely overfits (near-zero train error, high CV error) |
| **Random Forest** | Lowest, most stable CV error — selected |

### Hyperparameter tuning

`RandomizedSearchCV` (10 iterations, 3-fold CV, scored on negative RMSE)
tuned two parameters jointly:
- `preprocessing__geo__n_clusters` — number of k-means clusters for the
  geographic similarity features
- `random_forest__max_features` — features considered per split

Tuning both together matters because the right number of geo-clusters
depends on how many features the forest sees at each split, and vice versa.

### Final evaluation

The tuned pipeline is evaluated once on the held-out stratified test set
(never touched during training or tuning):

```python
final_predictions = final_model.predict(x_test)
final_rmse = root_mean_squared_error(y_test, final_predictions)
```

**Test RMSE ≈ $41,600** — meaning predictions are typically off by around
$41–42K on district median house values, which range roughly from
$15K–$500K in this dataset.

A bootstrap confidence interval (`scipy.stats.bootstrap`, 95% CI) on the
squared errors quantifies uncertainty in that RMSE estimate itself, rather
than treating it as a single fixed number.

### Reproducing training

```bash
cd backend
pip install -r requirements.txt
python train_model.py
```

This downloads the dataset, retrains the full pipeline (feature engineering
+ hyperparameter search + final fit), evaluates on the test set, and saves
the model to `backend/app/model/house_price_calculator_ml.pkl`.

The trained model is not committed to this repo (it's ~140MB — see
`.gitignore`); it's regenerated by the command above, and by the backend's
deploy build step in production.
backend/
├── app/
│ ├── ml_pipeline.py # ClusterSimilarity transformer (shared by train + serve)
│ ├── main.py # FastAPI app, exposes POST /predict
│ └── model/ # trained .pkl lands here after training
├── train_model.py # training entry point (see ML Training above)
└── requirements.txt
frontend/
└── index.html # static form, calls the backend directly

`ml_pipeline.py` exists as its own module specifically so the custom
`ClusterSimilarity` class is importable from the same path both when the
model is trained and pickled, and later when it's unpickled to serve
predictions — otherwise `joblib.load()` fails, since pickle stores a
reference to a class's import path rather than its code.

### Running locally

```bash
# backend
cd backend
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
python train_model.py
uvicorn app.main:app --reload --port 8000

# frontend — just open it
open frontend/index.html
```

### Deployment

- **Backend:** Render (free tier), Python 3.11, build command
  `pip install -r requirements.txt && python train_model.py`, start command
  `uvicorn app.main:app --host 0.0.0.0 --port $PORT`.
- **Frontend:** Netlify, static deploy of the `frontend/` folder.

## Tech stack

Python, scikit-learn, pandas, NumPy, SciPy, FastAPI, Uvicorn — HTML/CSS/JS
frontend with no build tooling.

---

## Application architecture
