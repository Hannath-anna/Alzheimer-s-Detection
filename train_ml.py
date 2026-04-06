"""
Alzheimer Detection - Machine Learning Training Script
Dataset: alzheimer_dataset.csv (Clinical/Tabular Data)
Model: Random Forest + XGBoost Ensemble
Outputs: trained model, scaler, label encoder, training graphs
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os, pickle, warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_curve, auc,
                             precision_recall_curve)
from sklearn.inspection import permutation_importance
import joblib

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
DATA_PATH   = "alzheimer_dataset.csv"
MODEL_DIR   = "models"
GRAPH_DIR   = "graphs"
RANDOM_SEED = 42
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(GRAPH_DIR, exist_ok=True)

np.random.seed(RANDOM_SEED)

# ─────────────────────────────────────────────
# 1. Load & Clean
# ─────────────────────────────────────────────
print("📂 Loading dataset...")
df = pd.read_csv(DATA_PATH)

# Drop non-predictive columns
drop_cols = ['PatientID', 'DoctorInCharge']
df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

print(f"   Shape: {df.shape}")
print(f"   Class distribution:\n{df['Diagnosis'].value_counts()}\n")

# ─────────────────────────────────────────────
# 2. Preprocessing
# ─────────────────────────────────────────────
X = df.drop('Diagnosis', axis=1)
y = df['Diagnosis']

feature_names = X.columns.tolist()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y)

print(f"   Train: {X_train.shape[0]} | Test: {X_test.shape[0]}\n")

# ─────────────────────────────────────────────
# 3. Train Multiple Models & Compare
# ─────────────────────────────────────────────
models = {
    "Random Forest":     RandomForestClassifier(n_estimators=200, max_depth=10,
                                                random_state=RANDOM_SEED, n_jobs=-1),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, learning_rate=0.05,
                                                     max_depth=5, random_state=RANDOM_SEED),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_SEED),
    "SVM":               SVC(kernel='rbf', probability=True, random_state=RANDOM_SEED)
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
cv_results = {}
trained_models = {}

print("🏋️  Training models with 5-Fold Cross-Validation...")
for name, model in models.items():
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
    cv_results[name] = scores
    model.fit(X_train, y_train)
    trained_models[name] = model
    print(f"   {name:22s} CV Acc: {scores.mean():.4f} ± {scores.std():.4f}")

# ─────────────────────────────────────────────
# 4. Best Model = Random Forest (main model)
# ─────────────────────────────────────────────
best_model = trained_models["Random Forest"]
y_pred     = best_model.predict(X_test)
y_prob     = best_model.predict_proba(X_test)[:, 1]

test_acc = accuracy_score(y_test, y_pred)
print(f"\n✅ Test Accuracy (Random Forest): {test_acc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["No Alzheimer", "Alzheimer"]))

# ─────────────────────────────────────────────
# 5. GRAPHS
# ─────────────────────────────────────────────
plt.style.use('seaborn-v0_8-darkgrid')
COLORS = ['#6C63FF', '#FF6584', '#43B89C', '#FFB347']

# --- 5a. Cross-Validation Accuracy Comparison ---
fig, ax = plt.subplots(figsize=(10, 5))
model_names = list(cv_results.keys())
means  = [cv_results[m].mean() for m in model_names]
stds   = [cv_results[m].std()  for m in model_names]
bars   = ax.bar(model_names, means, yerr=stds, color=COLORS, width=0.5,
                capsize=5, edgecolor='white', linewidth=1.2)
for bar, mean in zip(bars, means):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f'{mean:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
ax.set_ylim(0.7, 1.02)
ax.set_title('Cross-Validation Accuracy Comparison', fontsize=15, fontweight='bold', pad=15)
ax.set_ylabel('Accuracy', fontsize=12)
ax.set_xlabel('Model', fontsize=12)
plt.tight_layout()
plt.savefig(f'{GRAPH_DIR}/ml_model_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("   📊 Saved: ml_model_comparison.png")

# --- 5b. CV Fold Scores (line plot) ---
fig, ax = plt.subplots(figsize=(10, 5))
folds = [f"Fold {i+1}" for i in range(5)]
for idx, (name, scores) in enumerate(cv_results.items()):
    ax.plot(folds, scores, marker='o', linewidth=2, label=name, color=COLORS[idx])
ax.set_title('Cross-Validation Scores per Fold', fontsize=15, fontweight='bold', pad=15)
ax.set_ylabel('Accuracy', fontsize=12)
ax.set_xlabel('Fold', fontsize=12)
ax.legend(fontsize=10)
ax.set_ylim(0.7, 1.0)
plt.tight_layout()
plt.savefig(f'{GRAPH_DIR}/ml_cv_fold_scores.png', dpi=150, bbox_inches='tight')
plt.close()
print("   📊 Saved: ml_cv_fold_scores.png")

# --- 5c. Confusion Matrix ---
fig, ax = plt.subplots(figsize=(6, 5))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', ax=ax,
            xticklabels=["No Alzheimer", "Alzheimer"],
            yticklabels=["No Alzheimer", "Alzheimer"],
            linewidths=1, linecolor='white', cbar_kws={'shrink': 0.8})
ax.set_title('Confusion Matrix - Random Forest', fontsize=14, fontweight='bold', pad=12)
ax.set_ylabel('Actual', fontsize=12)
ax.set_xlabel('Predicted', fontsize=12)
plt.tight_layout()
plt.savefig(f'{GRAPH_DIR}/ml_confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.close()
print("   📊 Saved: ml_confusion_matrix.png")

# --- 5d. ROC Curve ---
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(fpr, tpr, color='#6C63FF', lw=2.5, label=f'ROC Curve (AUC = {roc_auc:.3f})')
ax.plot([0,1],[0,1], color='gray', linestyle='--', lw=1.5, label='Random Classifier')
ax.fill_between(fpr, tpr, alpha=0.1, color='#6C63FF')
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_title('ROC Curve - Random Forest', fontsize=14, fontweight='bold', pad=12)
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.legend(loc='lower right', fontsize=11)
plt.tight_layout()
plt.savefig(f'{GRAPH_DIR}/ml_roc_curve.png', dpi=150, bbox_inches='tight')
plt.close()
print("   📊 Saved: ml_roc_curve.png")

# --- 5e. Precision-Recall Curve ---
precision, recall, _ = precision_recall_curve(y_test, y_prob)
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(recall, precision, color='#FF6584', lw=2.5)
ax.fill_between(recall, precision, alpha=0.15, color='#FF6584')
ax.set_title('Precision-Recall Curve', fontsize=14, fontweight='bold', pad=12)
ax.set_xlabel('Recall', fontsize=12)
ax.set_ylabel('Precision', fontsize=12)
plt.tight_layout()
plt.savefig(f'{GRAPH_DIR}/ml_pr_curve.png', dpi=150, bbox_inches='tight')
plt.close()
print("   📊 Saved: ml_pr_curve.png")

# --- 5f. Feature Importance ---
importances = best_model.feature_importances_
indices = np.argsort(importances)[::-1][:20]
fig, ax = plt.subplots(figsize=(10, 7))
colors_feat = plt.cm.viridis(np.linspace(0.2, 0.9, 20))
bars = ax.barh([feature_names[i] for i in indices][::-1],
               importances[indices][::-1], color=colors_feat)
ax.set_title('Top 20 Feature Importances - Random Forest', fontsize=14, fontweight='bold', pad=12)
ax.set_xlabel('Importance Score', fontsize=12)
plt.tight_layout()
plt.savefig(f'{GRAPH_DIR}/ml_feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print("   📊 Saved: ml_feature_importance.png")

# --- 5g. Class Distribution ---
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
labels = ['No Alzheimer', 'Alzheimer']
counts = df['Diagnosis'].value_counts().sort_index().values
axes[0].pie(counts, labels=labels, autopct='%1.1f%%',
            colors=['#43B89C', '#FF6584'], startangle=90,
            wedgeprops={'edgecolor': 'white', 'linewidth': 2})
axes[0].set_title('Class Distribution', fontsize=13, fontweight='bold')
axes[1].bar(labels, counts, color=['#43B89C', '#FF6584'], edgecolor='white')
for i, c in enumerate(counts):
    axes[1].text(i, c + 10, str(c), ha='center', fontweight='bold', fontsize=12)
axes[1].set_title('Sample Count per Class', fontsize=13, fontweight='bold')
axes[1].set_ylabel('Count', fontsize=12)
plt.tight_layout()
plt.savefig(f'{GRAPH_DIR}/ml_class_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("   📊 Saved: ml_class_distribution.png")

# --- 5h. Correlation Heatmap (top features) ---
top_feats = [feature_names[i] for i in indices[:12]]
corr = df[top_feats + ['Diagnosis']].corr()
fig, ax = plt.subplots(figsize=(12, 9))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
            ax=ax, center=0, linewidths=0.5, annot_kws={'size': 8})
ax.set_title('Feature Correlation Heatmap (Top Features)', fontsize=14, fontweight='bold', pad=12)
plt.tight_layout()
plt.savefig(f'{GRAPH_DIR}/ml_correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("   📊 Saved: ml_correlation_heatmap.png")

# ─────────────────────────────────────────────
# 6. Save Model & Artifacts
# ─────────────────────────────────────────────
joblib.dump(best_model, f'{MODEL_DIR}/ml_model.pkl')
joblib.dump(scaler,     f'{MODEL_DIR}/ml_scaler.pkl')

artifacts = {
    'feature_names': feature_names,
    'test_accuracy': test_acc,
    'roc_auc': roc_auc,
    'class_names': ['No Alzheimer', 'Alzheimer']
}
with open(f'{MODEL_DIR}/ml_artifacts.pkl', 'wb') as f:
    pickle.dump(artifacts, f)

print(f"\n✅ Model saved  → {MODEL_DIR}/ml_model.pkl")
print(f"✅ Scaler saved → {MODEL_DIR}/ml_scaler.pkl")
print(f"✅ Training complete! Test Accuracy: {test_acc:.4f} | AUC: {roc_auc:.4f}")
