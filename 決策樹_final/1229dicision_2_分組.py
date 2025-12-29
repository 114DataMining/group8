import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    roc_curve, auc
)

# ============================================================
# 0) 基本設定：請改成你的檔案路徑與輸出資料夾
# ============================================================
FILE_PATH = r"C:\Users\User\Desktop\venv\test\dicision\1215dataset_new.csv"
TARGET_COL = "death"

OUTPUT_DIR = r"C:\Users\User\Desktop\venv\test\dicision\outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.2

# ============================================================
# 1) 讀資料 & 前處理
# ============================================================
df = pd.read_csv(FILE_PATH)

y = df[TARGET_COL]
X = df.drop(columns=[TARGET_COL])

# 若有類別欄位，轉 one-hot（決策樹可吃數值欄位）
X = pd.get_dummies(X, drop_first=True)

print(f"Loaded data: X shape={X.shape}, y positive rate={(y==1).mean():.3f}")

# ============================================================
# 2) 80/20 分層切分（final test set）
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    stratify=y,
    random_state=RANDOM_STATE
)

n_pos = int((y_train == 1).sum())
n_neg = int((y_train == 0).sum())
print(f"Train distribution: death=1 {n_pos}, death=0 {n_neg}, pos_rate={n_pos/(n_pos+n_neg):.3f}")

# ============================================================
# 3) 固定超參數訓練決策樹模型（你指定的那組）
# ============================================================
model = DecisionTreeClassifier(
    random_state=RANDOM_STATE,
    class_weight="balanced",   # 建議保留（不平衡資料）
    max_depth=3,
    min_samples_leaf=5,
    min_samples_split=2,
    criterion="entropy"
)

model.fit(X_train, y_train)

# ============================================================
# 4) 在 final test set 上輸出預測機率 + 指標（ACC/F1/AUC）
# ============================================================
y_pred = model.predict(X_test)
y_score = model.predict_proba(X_test)[:, 1]  # death=1 的機率

acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, zero_division=0)
auc_roc = roc_auc_score(y_test, y_score)

print("\n=== Final Test Performance (Fixed Hyperparams) ===")
print(f"Accuracy: {acc:.4f}")
print(f"F1      : {f1:.4f}")
print(f"AUC     : {auc_roc:.4f}")

# ============================================================
# 5) 產出決策樹圖（存檔，不 show）
# ============================================================
tree_path = os.path.join(OUTPUT_DIR, "decision_tree_fixed.png")

plt.figure(figsize=(22, 10))
plot_tree(
    model,
    feature_names=X_train.columns,
    class_names=["0", "1"],     # 可改成 ["Alive", "Death"]
    filled=True,
    rounded=True,
    max_depth=3,                # 與模型深度一致；若要只顯示前幾層可改 2~3
    fontsize=9
)
plt.title("Decision Tree (Fixed Hyperparams) - Top levels")
plt.tight_layout()
plt.savefig(tree_path, dpi=300)
plt.close()

print(f"Saved decision tree figure: {tree_path}")

# ============================================================
# 6) 畫 ROC curve + 算 AUC（存檔，不 show）
# ============================================================
roc_path = os.path.join(OUTPUT_DIR, "roc_curve_fixed_test.png")

fpr, tpr, thresholds = roc_curve(y_test, y_score)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(7, 6))
plt.plot(fpr, tpr, lw=2, label=f"ROC curve (AUC = {roc_auc:.3f})")
plt.plot([0, 1], [0, 1], lw=1, linestyle="--", label="Random guess")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve (Decision Tree, Final Test Set)")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(roc_path, dpi=300)
plt.close()

print(f"Saved ROC curve figure: {roc_path}")
print(f"ROC AUC (recomputed via curve): {roc_auc:.4f}")

# ==============
