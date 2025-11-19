import pickle
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import os
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("Data/train.csv", quoting=1)
df = df.dropna(subset=["headline"])

# Downsample each category to min_count
balanced_df = (
    df.groupby("category", group_keys=False)
      .apply(lambda x: x.sample(3000, random_state=42))
)

print("Original size:", len(df))
print("Balanced size:", len(balanced_df))

X = balanced_df["headline"]
y = balanced_df["category"]

le = LabelEncoder()
y_encoded = le.fit_transform(y) # for xgboost

pipelines = {
    "log_reg": make_pipeline(TfidfVectorizer(), LogisticRegression(max_iter=2000)),
    "naive_bayes": make_pipeline(TfidfVectorizer(), MultinomialNB()),
    "svm": make_pipeline(TfidfVectorizer(), SVC()),
    "random_forest": make_pipeline(TfidfVectorizer(), RandomForestClassifier()),
    "xgboost": make_pipeline(TfidfVectorizer(), xgb.XGBClassifier(eval_metric="mlogloss"))
}

param_grids = {
    "log_reg": {
        "logisticregression__C": [0.01, 0.1, 1, 3, 10],
        "logisticregression__penalty": ["l2"],
        "logisticregression__class_weight": [None, "balanced"],
        "logisticregression__max_iter": [300, 500, 800]
    },
    "naive_bayes": {
        "multinomialnb__alpha": [0.5, 1.0, 2.0]
    },
    "svm": {
        "svc__kernel": ["linear", "rbf"],
        "svc__C": [1, 10]
    },
    "random_forest": {
        "randomforestclassifier__n_estimators": [100, 300],
        "randomforestclassifier__max_depth": [None, 30]
    },
    "xgboost": {
        "xgbclassifier__n_estimators": [150, 300],
        "xgbclassifier__learning_rate": [0.05, 0.1]
    }
}

os.makedirs("Models/tuned", exist_ok=True)

summary_path = "Results/gridsearch_summary.csv"

if not os.path.exists(summary_path):
    pd.DataFrame(columns=["model", "best_score", "best_params"]).to_csv(summary_path, index=False)

for name, pipeline in pipelines.items():
    print(f"\n🔍 TUNING {name} ...")

    gs = GridSearchCV(
        pipeline,
        param_grids[name],
        cv=3,
        scoring="f1_macro",
        n_jobs=-1,
        verbose=3
    )

    gs.fit(X, y_encoded)
    
    best_params = gs.best_params_
    best_score = gs.best_score_

    print("BEST PARAMS:", best_params)
    print("BEST SCORE:", best_score)

    pickle.dump(gs.best_estimator_, open(f"Models/tuned/{name}.pkl", "wb"))
    
    row = pd.DataFrame([{
        "model": name,
        "best_score": best_score,
        "best_params": str(best_params)
    }])

    row.to_csv(summary_path, mode="a", index=False, header=False)
    
print("\nSummary saved to:", summary_path)