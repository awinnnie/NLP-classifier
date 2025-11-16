import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.pipeline import make_pipeline
import pickle
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import LabelEncoder 

print("Calling data")

train_df = pd.read_csv("Data\\train.csv", quoting = 1)
val_df = pd.read_csv("Data\\validation.csv", quoting = 1)

train_df = train_df.dropna(subset=["headline"]) # check nas
train_df = train_df[train_df["headline"] != ""]

val_df = val_df.dropna(subset=["headline"])
val_df = val_df[val_df["headline"] != ""]

print("Splitting the data")

X_train, y_train = train_df["headline"], train_df["category"]
X_val, y_val = val_df["headline"], val_df["category"]

tfidf = TfidfVectorizer(max_features=20000)
svd = TruncatedSVD(n_components=300)

models = {
    "log_reg": LogisticRegression(max_iter=1000),
    "naive_bayes": MultinomialNB(),
    "svm": LinearSVC(),
    "random_forest": RandomForestClassifier(n_estimators=100, random_state=42)
}

for name, model in models.items():
    print(f"Training {name}")
    if name == "naive_bayes":
        pipeline = make_pipeline(tfidf, model)
    else:
        pipeline = make_pipeline(tfidf, svd, model)
    pipeline.fit(X_train, y_train)
    pickle.dump(pipeline, open(f"Models/{name}.pkl", "wb"))
    
le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_val_enc = le.transform(y_val)

print("Training XGBoost")
xgb_pipeline = make_pipeline(
    TfidfVectorizer(max_features=20000),
    TruncatedSVD(n_components=300),
    xgb.XGBClassifier(eval_metric="mlogloss")  #multiclass
)

xgb_pipeline.fit(X_train, y_train_enc)
y_pred_xgb = xgb_pipeline.predict(X_val)
y_pred_xgb_labels = le.inverse_transform(y_pred_xgb)
pickle.dump(xgb_pipeline, open(f"Models/xgboost.pkl", "wb"))
