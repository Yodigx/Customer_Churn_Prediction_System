"""
Customer Churn Prediction - Industry-Grade Capstone Project
===========================================================
A full ML pipeline: data generation → EDA → preprocessing → 
model training → evaluation → visualization
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, accuracy_score, f1_score
)

# ─────────────────────────────────────────────────────────────────────────────
# 1.  GENERATE REALISTIC SYNTHETIC DATASET
# ─────────────────────────────────────────────────────────────────────────────
np.random.seed(42)
N = 2000

tenure        = np.random.randint(1, 72, N)
monthly       = np.round(np.random.uniform(18, 120, N), 2)
total_charges = np.round(monthly * tenure * np.random.uniform(0.85, 1.15, N), 2)
num_products  = np.random.choice([1, 2, 3, 4], N, p=[0.35, 0.35, 0.20, 0.10])
num_tickets   = np.random.poisson(1.2, N)
age           = np.random.randint(18, 75, N)
gender        = np.random.choice(["Male", "Female"], N)
contract      = np.random.choice(
    ["Month-to-month", "One year", "Two year"], N, p=[0.55, 0.25, 0.20]
)
payment       = np.random.choice(
    ["Electronic check", "Mailed check", "Bank transfer", "Credit card"], N
)
internet      = np.random.choice(
    ["DSL", "Fiber optic", "No"], N, p=[0.35, 0.45, 0.20]
)
tech_support  = np.random.choice(["Yes", "No"], N, p=[0.35, 0.65])

# Churn probability shaped by business logic
churn_prob = (
    0.05
    + 0.20 * (contract == "Month-to-month")
    - 0.10 * (contract == "Two year")
    + 0.15 * (num_tickets > 2)
    - 0.08 * (num_products > 2)
    - 0.005 * (tenure / 12)
    + 0.10 * (internet == "Fiber optic")
    - 0.05 * (tech_support == "Yes")
    + 0.08 * (payment == "Electronic check")
    + np.random.normal(0, 0.05, N)
)
churn_prob = np.clip(churn_prob, 0.01, 0.99)
churn      = (np.random.rand(N) < churn_prob).astype(int)

df = pd.DataFrame({
    "CustomerID":    [f"CUST{i:04d}" for i in range(1, N+1)],
    "Gender":        gender,
    "Age":           age,
    "Tenure":        tenure,
    "Contract":      contract,
    "InternetService": internet,
    "TechSupport":   tech_support,
    "PaymentMethod": payment,
    "NumProducts":   num_products,
    "NumSupportTickets": num_tickets,
    "MonthlyCharges":    monthly,
    "TotalCharges":      total_charges,
    "Churn":             churn,
})

df.to_csv("churn_dataset.csv", index=False)
print(f"Dataset shape : {df.shape}")
print(f"Churn rate    : {df['Churn'].mean():.2%}\n")

# ─────────────────────────────────────────────────────────────────────────────
# 2.  EXPLORATORY DATA ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
print("=== EDA Summary ===")
print(df.describe().to_string())
print("\nMissing values:\n", df.isnull().sum())

# ─────────────────────────────────────────────────────────────────────────────
# 3.  PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────
df_model = df.drop(columns=["CustomerID"]).copy()

# Encode categoricals
le        = LabelEncoder()
cat_cols  = ["Gender", "Contract", "InternetService", "TechSupport", "PaymentMethod"]
for c in cat_cols:
    df_model[c] = le.fit_transform(df_model[c])

X = df_model.drop(columns=["Churn"])
y = df_model["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# ─────────────────────────────────────────────────────────────────────────────
# 4.  MODEL TRAINING
# ─────────────────────────────────────────────────────────────────────────────
models = {
    "Logistic Regression":    LogisticRegression(max_iter=500, random_state=42),
    "Random Forest":          RandomForestClassifier(n_estimators=150, random_state=42),
    "Gradient Boosting":      GradientBoostingClassifier(n_estimators=150, random_state=42),
}

cv       = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results  = {}

print("\n=== Cross-Validation Results ===")
for name, model in models.items():
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc")
    model.fit(X_train, y_train)
    y_pred      = model.predict(X_test)
    y_prob      = model.predict_proba(X_test)[:, 1]
    results[name] = {
        "model":    model,
        "y_pred":   y_pred,
        "y_prob":   y_prob,
        "cv_mean":  cv_scores.mean(),
        "cv_std":   cv_scores.std(),
        "accuracy": accuracy_score(y_test, y_pred),
        "f1":       f1_score(y_test, y_pred),
        "auc":      roc_auc_score(y_test, y_prob),
    }
    print(f"{name:25s}  CV AUC={cv_scores.mean():.4f}±{cv_scores.std():.4f}  "
          f"Test AUC={results[name]['auc']:.4f}  Acc={results[name]['accuracy']:.4f}")

best_name  = max(results, key=lambda n: results[n]["auc"])
best       = results[best_name]
print(f"\nBest model: {best_name}  (AUC={best['auc']:.4f})")
print("\nClassification Report:\n", classification_report(y_test, best["y_pred"],
      target_names=["No Churn", "Churn"]))

# ─────────────────────────────────────────────────────────────────────────────
# 5.  VISUALISATION DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────
plt.style.use("seaborn-v0_8-whitegrid")
PALETTE = {"churn": "#E05C5C", "no_churn": "#5B8FF9", "accent": "#38BFA1"}
fig     = plt.figure(figsize=(20, 16), facecolor="#F8F9FB")
fig.suptitle("Customer Churn Prediction — Project Dashboard",
             fontsize=22, fontweight="bold", y=0.98, color="#1C1C2E")

gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.42, wspace=0.35)

# — Churn Distribution —
ax0 = fig.add_subplot(gs[0, 0])
counts = df["Churn"].value_counts()
bars   = ax0.bar(["No Churn", "Churn"], counts.values,
                 color=[PALETTE["no_churn"], PALETTE["churn"]], width=0.5, edgecolor="white")
for bar, v in zip(bars, counts.values):
    ax0.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
             f"{v}\n({v/N:.1%})", ha="center", va="bottom", fontsize=10, fontweight="bold")
ax0.set_title("Churn Distribution", fontweight="bold")
ax0.set_ylabel("Customers")
ax0.set_ylim(0, max(counts.values) * 1.2)

# — Churn by Contract Type —
ax1 = fig.add_subplot(gs[0, 1])
churn_contract = df.groupby("Contract")["Churn"].mean() * 100
churn_contract.sort_values().plot(kind="barh", ax=ax1,
                                   color=[PALETTE["no_churn"], PALETTE["accent"], PALETTE["churn"]])
ax1.set_title("Churn Rate by Contract Type", fontweight="bold")
ax1.set_xlabel("Churn Rate (%)")
for bar in ax1.patches:
    ax1.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
             f"{bar.get_width():.1f}%", va="center", fontsize=9)

# — Monthly Charges Distribution —
ax2 = fig.add_subplot(gs[0, 2])
for label, color, mask in [
    ("No Churn", PALETTE["no_churn"], df["Churn"] == 0),
    ("Churn",    PALETTE["churn"],    df["Churn"] == 1),
]:
    ax2.hist(df.loc[mask, "MonthlyCharges"], bins=30, alpha=0.65,
             label=label, color=color, edgecolor="white")
ax2.set_title("Monthly Charges by Churn", fontweight="bold")
ax2.set_xlabel("Monthly Charges ($)")
ax2.legend(fontsize=9)

# — ROC Curves —
ax3 = fig.add_subplot(gs[1, 0:2])
colors_roc = [PALETTE["churn"], PALETTE["accent"], PALETTE["no_churn"]]
for (name, res), clr in zip(results.items(), colors_roc):
    fpr, tpr, _ = roc_curve(y_test, res["y_prob"])
    ax3.plot(fpr, tpr, label=f"{name} (AUC={res['auc']:.3f})", color=clr, lw=2)
ax3.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
ax3.set_title("ROC Curves — Model Comparison", fontweight="bold")
ax3.set_xlabel("False Positive Rate")
ax3.set_ylabel("True Positive Rate")
ax3.legend(fontsize=9)

# — Confusion Matrix (best model) —
ax4 = fig.add_subplot(gs[1, 2])
cm = confusion_matrix(y_test, best["y_pred"])
sns.heatmap(cm, annot=True, fmt="d", ax=ax4,
            cmap=sns.light_palette(PALETTE["churn"], as_cmap=True),
            xticklabels=["No Churn", "Churn"],
            yticklabels=["No Churn", "Churn"], linewidths=0.5)
ax4.set_title(f"Confusion Matrix\n{best_name}", fontweight="bold")
ax4.set_ylabel("Actual")
ax4.set_xlabel("Predicted")

# — Feature Importance (best model if RF/GB) —
ax5 = fig.add_subplot(gs[2, 0:2])
best_model = best["model"]
feat_names = X.columns.tolist()
if hasattr(best_model, "feature_importances_"):
    importances = best_model.feature_importances_
else:
    importances = np.abs(best_model.coef_[0])
fi = pd.Series(importances, index=feat_names).sort_values(ascending=True)
colors_fi = [PALETTE["churn"] if v >= fi.quantile(0.75) else PALETTE["no_churn"] for v in fi]
fi.plot(kind="barh", ax=ax5, color=colors_fi)
ax5.set_title(f"Feature Importances — {best_name}", fontweight="bold")
ax5.set_xlabel("Importance Score")

# — Model Metric Comparison —
ax6 = fig.add_subplot(gs[2, 2])
metric_df = pd.DataFrame({
    n: {"Accuracy": r["accuracy"], "F1 Score": r["f1"], "ROC-AUC": r["auc"]}
    for n, r in results.items()
}).T
x    = np.arange(len(metric_df))
w    = 0.25
cols = [PALETTE["no_churn"], PALETTE["accent"], PALETTE["churn"]]
for i, (col, metric) in enumerate(zip(cols, metric_df.columns)):
    ax6.bar(x + i*w, metric_df[metric], width=w, label=metric, color=col, alpha=0.85)
ax6.set_xticks(x + w)
ax6.set_xticklabels(["LR", "RF", "GB"], fontsize=9)
ax6.set_ylim(0.5, 1.0)
ax6.set_title("Model Metrics Comparison", fontweight="bold")
ax6.legend(fontsize=8)
ax6.set_ylabel("Score")

plt.savefig("churn_dashboard.png", dpi=150,
            bbox_inches="tight", facecolor=fig.get_facecolor())
print("\n✅ Dashboard saved to churn_dashboard.png")

# ─────────────────────────────────────────────────────────────────────────────
# 6.  EXPORT PREDICTIONS
# ─────────────────────────────────────────────────────────────────────────────
test_ids  = df.iloc[y_test.index]["CustomerID"].values
pred_df   = pd.DataFrame({
    "CustomerID":      test_ids,
    "Actual_Churn":    y_test.values,
    "Predicted_Churn": best["y_pred"],
    "Churn_Probability": np.round(best["y_prob"], 4),
    "Risk_Segment": pd.cut(best["y_prob"], bins=[0, 0.3, 0.6, 1.0],
                           labels=["Low", "Medium", "High"]),
})
pred_df.to_csv("churn_predictions.csv", index=False)
print("✅ Predictions saved to churn_predictions.csv")
print(pred_df.head(10).to_string(index=False))
