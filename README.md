# News Headline Classifier

A machine learning system that classifies news headlines into **15 categories** using a tuned Support Vector Machine (SVM). The project includes a full ML pipeline, a FastAPI backend, and an interactive Streamlit frontend — all containerized with Docker and Docker-compose.

**Categories:** `COMEDY` · `PARENTING` · `SPORTS` · `ENTERTAINMENT` · `POLITICS` · `WELLNESS` · `BUSINESS` · `STYLE & BEAUTY` · `FOOD & DRINK` · `QUEER VOICES` · `HOME & LIVING` · `BLACK VOICES` · `TRAVEL` · `PARENTS` · `HEALTHY LIVING`

---

## Table of Contents

- [Project Overview](#project-overview)
- [Project Structure](#project-structure)
- [Machine Learning](#machine-learning)
- [API Endpoints](#api-endpoints)
- [Setup & Installation](#setup--installation)
  - [Running with Docker (Recommended)](#running-with-docker-recommended)
  - [Running Locally](#running-locally)
- [Usage](#usage)

---

## Project Overview

This project takes a news headline as input and predicts which of 15 editorial categories it belongs to. It was built using the [HuffPost News Category Dataset (v3)](https://www.kaggle.com/datasets/rmisra/news-category-dataset).

The system has three main components:

| Component | Technology | Description |
|---|---|---|
| **ML Pipeline** | scikit-learn, XGBoost | Model training, evaluation, and hyperparameter tuning |
| **Backend API** | FastAPI |API for predictions and data management |
| **Frontend** | Streamlit | Interactive UI for prediction and data analysis |

---

## Project Structure

```
├── backend/
│   ├── api/
│   │   ├── main.py                  # FastAPI app entry point
│   │   └── routers/
│   │       ├── prediction.py        # /predict endpoint
│   │       └── data_analysis.py     # CRUD endpoints for headlines
│   ├── ml/
│   │   ├── split.py                 # Dataset splitting (train/validation/test)
│   │   ├── train-test.py            # Baseline model training
│   │   ├── evaluate.py              # Baseline model evaluation
│   │   ├── Gridsearchcv.py          # Hyperparameter tuning with GridSearchCV
│   │   ├── eval_tuned.py            # Tuned model evaluation
│   │   └── checking.py              # Category distribution inspection
│   ├── Models/
│   │   ├── log_reg.pkl              # Baseline models (not included in the GitHub repository)
│   │   ├── naive_bayes.pkl
│   │   ├── svm.pkl
│   │   ├── xgboost.pkl
│   │   └── tuned/
│   │       ├── log_reg.pkl          # Tuned models (GridSearchCV best estimators)
│   │       ├── naive_bayes.pkl
│   │       └── svm.pkl              # used in production
│   ├── Data/
│   │   ├── News_Category_Dataset_v3.json   # Raw HuffPost dataset
│   │   ├── train.csv                       # 118,497 rows
│   │   ├── validation.csv                  # 14,812 rows
│   │   └── test.csv                        # 14,813 rows
│   ├── Results/
│   │   ├── metrics.csv                     # Baseline model metrics
│   │   ├── best_metrics.csv                # Tuned model metrics
│   │   ├── gridsearch_summary.csv          # GridSearchCV best params + scores
│   │   ├── confusion_matrices/             # Baseline confusion matrix plots
│   │   ├── roc_curves/                     # Baseline ROC curve plots
│   │   ├── best_confusion_matrices/        # Tuned confusion matrix plots
│   │   └── best_roc_curves/                # Tuned ROC curve plots
│   ├── db.py                        # SQLAlchemy engine and session
│   ├── models.py                    # NewsRow ORM model (id, headline, category)
│   ├── seed_db.py                   # Seeds SQLite DB from raw JSON (skips if already seeded)
│   └── news.db                      # SQLite database
├── frontend/
│   ├── main.py                      # Streamlit entry point + API base URL config
│   └── pages/
│       ├── Prediction.py            # Headline prediction UI
│       ├── Data_Analysis.py         # Category counts and sample headlines
│       └── Data_Manipulation.py     # Add / update / delete headlines
├── docker-compose.yml
└── README.md
```

---

## Machine Learning

### Dataset

The top 15 categories by frequency were selected from the HuffPost dataset and split stratified by category:

| Split | Size |
|---|---|
| Train | 118,497 |
| Validation | 14,812 |
| Test | 14,813 |

### Pipeline

All models use a `TfidfVectorizer / Classifier` scikit-learn pipeline. Five classifiers were trained and evaluated at baseline: **Logistic Regression**, **Naive Bayes**, **SVM** (LinearSVC for baseline, SVC for tuning), **Random Forest** and **XGBoost**.

Hyperparameter tuning was performed using `GridSearchCV` (3-fold CV, `f1_macro` scoring) on a class-balanced subsample of 3,000 rows per category. Three models were tuned and saved: Logistic Regression, Naive Bayes and SVM.

### Selected model: Tuned SVM

I have selected the Tuned SVM model after comparing it against Logistic Regression, Naive Bayes, XGBoost and Random Forest. SVM achieved the best overall F1-score, and it's also well-suited for high-dimensional sparse data like TF-IDF vectors, which is typical in text classification. The tuned SVM model is loaded at startup by both the prediction and data analysis routers:

```
backend/Models/tuned/svm.pkl
```

### Evaluation Outputs

All results are saved to `backend/Results/`:

| File / Folder | Contents |
|---|---|
| `metrics.csv` | Accuracy, Precision, Recall, F1 for baseline models |
| `best_metrics.csv` | Same metrics for tuned models |
| `gridsearch_summary.csv` | Best params and CV score per model |
| `confusion_matrices/` | Per-model confusion matrix heatmaps (baseline) |
| `best_confusion_matrices/` | Per-model confusion matrix heatmaps (tuned) |
| `roc_curves/` | Per-category ROC curves, one-vs-rest (baseline) |
| `best_roc_curves/` | Per-category ROC curves, one-vs-rest (tuned) |

---

## API Endpoints

The FastAPI backend runs on `http://localhost:8000`. Interactive docs are available at `http://localhost:8000/docs`.

### Prediction

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/predict` | Predict the category of a headline |

**Request:**
```json
{ "text": "Scientists discover new species in the Amazon" }
```

**Response:**
```json
{ "prediction": "TRAVEL" }
```

---

### Data Analysis

All write operations (add, update) automatically re-classify the headline through the tuned SVM. The stored category always reflects the model's prediction.

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/count/{category}` | Count of headlines in a category |
| `GET` | `/top10/{category}` | First 10 headlines in a category (ordered by ID) |
| `POST` | `/add_headline` | Add a new headline; category is auto-predicted |
| `PUT` | `/update_headline` | Update headline text by ID; category is re-predicted |
| `DELETE` | `/delete_headline` | Delete a headline by ID |

**Example — Add headline:**
```json
POST /add_headline
{ "text": "Best smoothie recipes for summer" }
```
```json
{
  "message": "Headline added",
  "id": 42,
  "headline": "Best smoothie recipes for summer",
  "predicted_category": "FOOD & DRINK"
}
```

**Example — Update headline:**
```json
PUT /update_headline
{ "id": 42, "text": "Top travel destinations in 2025" }
```
```json
{
  "message": "Headline updated",
  "id": 42,
  "new_headline": "Top travel destinations in 2025",
  "new_category": "TRAVEL"
}
```

**Example — Delete headline:**
```json
DELETE /delete_headline
{ "id": 42 }
```
```json
{ "message": "Headline deleted", "id": 42 }
```

---

## Setup & Installation

### Prerequisites

- [Docker](https://www.docker.com/) and Docker Compose,
  **or**
- Python 3.10+

---

### Running with Docker 

The `docker-compose.yml` defines two services: `backend` (FastAPI on port 8000) and `frontend` (Streamlit on port 8501). The frontend reaches the backend via the internal Docker network (`API_BASE=http://backend:8000`).

1. **Clone the repository:**
   ```bash
   git clone <repo-url>
   cd <repo-folder>
   ```

2. **Build and start all services:**
   ```bash
   docker-compose up --build
   ```

3. **Access the applications:**
   - Frontend (Streamlit): [http://localhost:8501](http://localhost:8501)
   - Backend API: [http://localhost:8000](http://localhost:8000)
   - API Docs (Swagger): [http://localhost:8000/docs](http://localhost:8000/docs)

4. **Stop the services:**
   ```bash
   docker-compose down
   ```

> **Note:** The database is seeded automatically on first startup from `News_Category_Dataset_v3.json`. If the database already contains rows, seeding is skipped.

---

### Running Locally

#### Backend

```bash
cd backend

# Install dependencies
pip install -r requirements_backend.txt

# Seed the database (first run only)
python seed_db.py

# Start the API
uvicorn api.main:app --reload --port 8000
```

#### Frontend

Open a second terminal:

```bash
cd frontend

# Install dependencies
pip install -r requirements_frontend.txt

# Start the Streamlit app
streamlit run main.py
```

---

## How to use

Navigate to the Streamlit frontend at `http://localhost:8501`.

The API base URL is configurable in the **sidebar** of the app (defaults to `http://127.0.0.1:8000` locally, or the Docker internal URL when containerized).

- **Prediction** — Enter any news headline and get an instant category prediction from the tuned SVM.
- **Data Analysis** — Select a category to see its headline count or browse the first 10 entries.
- **Data Manipulation** — Add new headlines (auto-classified), update existing ones by ID, or delete them by ID.

