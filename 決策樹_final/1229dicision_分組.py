import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# ============================================================
# 0) 基本設定
# ============================================================
FILE_PATH = r"C:\Users\User\Desktop\venv\test\dicision\1215dataset_new.csv"
TARGET_COL = "death"

RANDOM_STATE = 42
N_SPLITS = 5

# ============================================================
# 1) 讀資料
# ============================================================
df = pd.read_csv(FILE_PATH)
y = df[TARGET_COL]
X = df.drop(columns=[TARGET_COL])

X = pd.get_dummies(X, drop_first=True)

print(f"Loaded data: X shape={X.shape}, y positive rate={(y==1).mean():.3f}")

# ============================================================
# 2) 80/20 切分（分層）
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)

n_pos = int((y_train == 1).sum())
n_neg = int((y_train == 0).sum())
print(f"\nTrain distribution: death=1 {n_pos}, death=0 {n_neg}, pos_rate={n_pos/(n_pos+n_neg):.3f}")

# ============================================================
# 3) 5-fold + GridSearchCV（收斂）
# ============================================================
cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

base_model = DecisionTreeClassifier(
    random_state=RANDOM_STATE,
    class_weight="balanced"
)

param_grid = {
    "max_depth": [2, 3, 4, 5, 6, 8, 10, None],
    "min_samples_split": [2, 5, 10, 20],
    "min_samples_leaf": [1, 2, 5, 10],
    "criterion": ["gini", "entropy"]
}

grid = GridSearchCV(
    estimator=base_model,
    param_grid=param_grid,
    scoring="roc_auc",
    cv=cv,
    n_jobs=-1
)

grid.fit(X_train, y_train)

best_params = grid.best_params_
best_model = grid.best_estimator_

print("\n=== Converged (Selected) Hyperparameters ===")
print("Best params:", best_params)
print("Best CV AUC:", grid.best_score_)

# ============================================================
# 4) Final test 評估（20% hold-out）
# ============================================================
y_pred_test = best_model.predict(X_test)
y_score_test = best_model.predict_proba(X_test)[:, 1]

acc_test = accuracy_score(y_test, y_pred_test)
f1_test = f1_score(y_test, y_pred_test, zero_division=0)
auc_test = roc_auc_score(y_test, y_score_test)

print("\n=== Final Test Performance (20% hold-out) ===")
print(f"Accuracy: {acc_test:.4f}")
print(f"F1      : {f1_test:.4f}")
print(f"AUC     : {auc_test:.4f}")

# ============================================================
# 5) 畫決策樹（存檔，不 show）
# ============================================================
plt.figure(figsize=(22, 10))
plot_tree(
    best_model,
    feature_names=X_train.columns,
    class_names=["0", "1"],
    filled=True,
    rounded=True,
    max_depth=3,
    fontsize=9
)
plt.title("Decision Tree (best params) - Top levels")
plt.tight_layout()
plt.savefig("decision_tree.png", dpi=300)
plt.close()

print("Saved decision tree to decision_tree.png")

# ============================================================
# 6) 固定 best_params 重跑 5-fold（每折 ACC / F1 / AUC）
# ============================================================
model_cv = DecisionTreeClassifier(
    random_state=RANDOM_STATE,
    class_weight="balanced",
    **best_params
)

skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

accs, f1s, aucs = [], [], []

print("\n=== 5-fold results (per fold) ===")
for fold, (tr_idx, te_idx) in enumerate(skf.split(X, y), start=1):
    X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
    y_tr, y_te = y.iloc[tr_idx], y.iloc[te_idx]

    n_pos_fold = int((y_tr == 1).sum())
    n_neg_fold = int((y_tr == 0).sum())
    pos_rate_fold = n_pos_fold / (n_pos_fold + n_neg_fold)

    model_cv.fit(X_tr, y_tr)

    y_pred = model_cv.predict(X_te)
    y_score = model_cv.predict_proba(X_te)[:, 1]

    acc = accuracy_score(y_te, y_pred)
    f1 = f1_score(y_te, y_pred, zero_division=0)
    auc = roc_auc_score(y_te, y_score)

    accs.append(acc)
    f1s.append(f1)
    aucs.append(auc)

    print(f"Fold {fold}: pos_rate(train)={pos_rate_fold:.3f} | "
          f"ACC={acc:.4f} | F1={f1:.4f} | AUC={auc:.4f}")

print("\n=== 5-fold summary (mean ± std) ===")
print(f"ACC: {np.mean(accs):.4f} ± {np.std(accs, ddof=1):.4f}")
print(f"F1 : {np.mean(f1s):.4f} ± {np.std(f1s, ddof=1):.4f}")
print(f"AUC: {np.mean(aucs):.4f} ± {np.std(aucs, ddof=1):.4f}")
