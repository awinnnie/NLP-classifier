import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import xgboost as xgb
from sklearn.pipeline import make_pipeline


train_df = pd.read_csv("train.csv", quoting = 1)
val_df = pd.read_csv("validation.csv", quoting = 1)

train_df = train_df.dropna(subset=["headline"])
train_df = train_df[train_df["headline"] != ""]

val_df = val_df.dropna(subset=["headline"])
val_df = val_df[val_df["headline"] != ""]

X_train, y_train = train_df["headline"], train_df["category"]
X_val, y_val = val_df["headline"], val_df["category"]


tfidf = TfidfVectorizer()
X_train_tfidf = tfidf.fit_transform(X_train)
X_val_tfidf = tfidf.transform(X_val)

#log reg

lr_pipeline = LogisticRegression(max_iter=1000).fit(X_train_tfidf, y_train)
y_pred_lr = lr_pipeline.predict(X_val)

log_acc = accuracy_score(y_val, y_pred_lr)
log_precision = precision_score(y_val, y_pred_lr, average="macro")
log_recall = recall_score(y_val, y_pred_lr, average="macro")
log_f1 = f1_score(y_val, y_pred_lr, average="macro")

# print(classification_report(y_val, y_pred_lr))

#naive bayes

nb_pipeline = MultinomialNB.fit(X_train_tfidf, y_train)
y_pred_nb = nb_pipeline.predict(X_val)

nb_acc = accuracy_score(y_val, y_pred_nb)
nb_precision = precision_score(y_val, y_pred_nb, average="macro")
nb_recall = recall_score(y_val, y_pred_nb, average="macro")
nb_f1 = f1_score(y_val, y_pred_nb, average="macro")


#SVM

svm_pipeline = SVC(kernel="linear").fit(X_train_tfidf, y_train)
y_pred_svm = svm_pipeline.predict(X_val)

svm_acc = accuracy_score(y_val, y_pred_svm)
svm_precision = precision_score(y_val, y_pred_svm, average="macro")
svm_recall = recall_score(y_val, y_pred_svm, average="macro")
svm_f1 = f1_score(y_val, y_pred_svm, average="macro")


#random forest

rf_pipeline = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train_tfidf, y_train)
y_pred_rf = rf_pipeline.predict(X_val)

rf_acc = accuracy_score(y_val, y_pred_rf)
rf_precision = precision_score(y_val, y_pred_rf, average="macro")
rf_recall = recall_score(y_val, y_pred_rf, average="macro")
rf_f1 = f1_score(y_val, y_pred_rf, average="macro")


#XGBoost

metrics = {
    "Model": ["Logistic Regression", "Naive Bayes", "SVM", "Random Forest"],
    "Accuracy": [log_acc, nb_acc, svm_acc, rf_acc],
    "Precision": [log_precision, nb_precision, svm_precision, rf_precision],
    "Recall": [log_recall, nb_recall, svm_recall, rf_recall],
    "F1-score": [log_f1, nb_f1, svm_f1, rf_f1]
}

metrics_df = pd.DataFrame(metrics)
print(metrics_df)