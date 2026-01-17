import pandas as pd
import pickle
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

val_df = pd.read_csv("Data\\validation.csv", quoting=1)
val_df = val_df.dropna(subset=["headline"])
val_df = val_df[val_df["headline"] != ""]

X_val = val_df["headline"]
y_val = val_df["category"]

le = LabelEncoder()
le.fit(pd.read_csv("Data/train.csv", quoting=1)["category"])

y_val_enc = le.transform(y_val)
y_val_bin = label_binarize(y_val_enc, classes=range(len(le.classes_)))

results = []

for file in os.listdir("Models/tuned"):
    if file.endswith(".pkl"):
        name = file.replace(".pkl", "")
        pipeline = pickle.load(open(os.path.join("Models/tuned", file), "rb"))
        
        y_pred_enc = pipeline.predict(X_val)
        y_pred = le.inverse_transform(y_pred_enc)
        
        # ROC
        if hasattr(pipeline, "predict_proba"):
            y_score = pipeline.predict_proba(X_val)
        else:
            y_score = pipeline.decision_function(X_val)
            if y_score.ndim == 1:
                y_score = y_score.reshape(-1, 1)
        # else:
        #     y_pred = pipeline.predict(X_val)
        #     #check if model supports predict_proba
        #     if hasattr(pipeline, "predict_proba"):
        #         y_score = pipeline.predict_proba(X_val)
        #     else:
        #         y_score = pipeline.decision_function(X_val)
        #         #If 1d for binary, reshape
        #         if y_score.ndim == 1:
        #             y_score = y_score.reshape(-1,1)

        # Metrics
        acc = accuracy_score(y_val, y_pred)
        prec = precision_score(y_val, y_pred, average="macro")
        rec = recall_score(y_val, y_pred, average="macro")
        f1 = f1_score(y_val, y_pred, average="macro")

        results.append({
            "Model": name,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1-score": f1
        })

        # Confusion matrix
        cm = confusion_matrix(y_val, y_pred, labels=le.classes_)
        plt.figure(figsize=(10,8))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=le.classes_, yticklabels=le.classes_)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title(f"Confusion Matrix: {name}")
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(os.path.join("Results/best_confusion_matrices", f"{name}_cm.png"))
        plt.close()
        
        # ROC 1 vs rest
        n_classes = len(le.classes_)
        plt.figure(figsize=(10,8))
        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(y_val_bin[:,i], y_score[:,i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=2, label=f'{le.classes_[i]} (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curves: {name}')
        plt.legend(loc="lower right", fontsize='small', ncol=2)
        plt.tight_layout()
        plt.savefig(os.path.join("Results/best_roc_curves", f"{name}_roc.png"))
        plt.close()
        
results_df = pd.DataFrame(results).sort_values("Accuracy", ascending=False)
results_df.to_csv(os.path.join("Results", "best_metrics.csv"), index=False)
print("\nModel Performance\n")
print(results_df.to_string(index=False))
print(f"\nMetrics saved")
print(f"Confusion matrices saved")
print(f"ROC curves saved")