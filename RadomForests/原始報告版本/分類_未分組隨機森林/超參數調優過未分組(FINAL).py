import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score, f1_score, make_scorer, roc_curve
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import resample

# 匯入數據
try:
    df_test = pd.read_csv('heartattack_test_20_CLF.csv')
    df_train = pd.read_csv('heartattack_train_80_CLF.csv')
    print("數據匯入成功")
except FileNotFoundError:
    print("錯誤：找不到 CLF 檔案")
    exit()

feature_cols = ['Zejectionfraction', 'Zplatelets', 'Zserumsodium','Zserumcreatinine', 'Zcreatininephosphokinase', 'age_new','anaemia', 'diabetes', 'highbp', 'sex', 'smoking'
]
target_col = 'death'

X_train = df_train[feature_cols]
y_train = df_train[target_col]
X_test = df_test[feature_cols]
y_test = df_test[target_col]

param_grid = {
'n_estimators': [200, 300],
'max_depth': [4, 6, 8],
'min_samples_leaf': [2, 3],
'class_weight': [None, 'balanced'],
'max_features': ['sqrt', 'log2']
}

# 定義評估標準
f1_scorer = make_scorer(f1_score, pos_label=1)

# 執行
rf_base = RandomForestClassifier(random_state=42)

grid_search = GridSearchCV(estimator=rf_base,
param_grid=param_grid,
scoring=f1_scorer,
cv=5,
verbose=2,
n_jobs=-1
)

print("\n開始 Grid Search")
grid_search.fit(X_train, y_train)

# 結果分析
print("\n--- Grid Search 結果 ---")
print(f"參數組合: {grid_search.best_params_}")
print(f"交叉驗證的F1-Score 分數: {grid_search.best_score_:.4f}")

best_rf = grid_search.best_estimator_
y_pred_tuned = best_rf.predict(X_test)
y_prob_tuned = best_rf.predict_proba(X_test)[:, 1]

# 計算最終評估指標
f1_tuned = f1_score(y_test, y_pred_tuned, pos_label=1)
report_tuned = classification_report(y_test, y_pred_tuned)
auc_tuned = roc_auc_score(y_test, y_prob_tuned)
accuracy_tuned = grid_search.best_estimator_.score(X_test, y_test)

print("\n---最佳模型在測試集上的表現---")
print(f"整體準確度 (Accuracy): {accuracy_tuned:.4f}")
print(f"F1-Score (Death=1): {f1_tuned:.4f} (平衡指標)")
print(f"AUC: {auc_tuned:.4f}")
print("\n分類報告 (Classification Report):")
print(report_tuned)

from sklearn.tree import plot_tree

first_tree = best_rf.estimators_[0]

plt.figure(figsize=(25, 12)) 
plot_tree(
    first_tree, 
    feature_names=feature_cols, 
    class_names=['Survived (0)', 'Death (1)'], 
    filled=True,      
    rounded=True,     
    precision=2,      
    max_depth=3       
)

plt.title(f"隨機森林中的第一棵決策樹展示\n(使用最佳參數: {grid_search.best_params_})", fontsize=16)
plt.show()

print("\n--- 隨機森林內部結構拆解 ---")

n_samples = X_train.shape[0]
n_features = X_train.shape[1]
print(f"原始訓練集總樣本數 (N): {n_samples}")
print(f"原始特徵總數 (M): {n_features}")

# 每一棵樹的樣本大小與特徵數

print(f"每棵樹抽取的樣本數 (Bootstrap Sample Size): {n_samples}")

# best_params_ ('max_features': 'sqrt')
import math
max_features_val = math.floor(math.sqrt(n_features))
print(f"每棵樹在每個節點隨機挑選的特徵數 (max_features='sqrt'): {max_features_val}")

# 檢查森林中的第一棵樹
first_tree = best_rf.estimators_[0]
print(f"\n[第一棵樹的抽樣特徵]")
print(f"這棵樹的節點總數: {first_tree.tree_.node_count}")
print(f"這棵樹的最大深度: {first_tree.get_depth()}")

# min_samples_leaf
all_leaves_samples = []
for tree in best_rf.estimators_:
    n_node_samples = tree.tree_.n_node_samples
    children_left = tree.tree_.children_left
    leaves_samples = n_node_samples[children_left == -1]
    all_leaves_samples.extend(leaves_samples)

print(f"所有樹的葉節點中，最小的樣本數: {min(all_leaves_samples)} (應符合 min_samples_leaf=3)")

# 視覺化 ROC 曲線 
try:
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
except NameError:
    pass

sns.set_style("whitegrid")
fpr, tpr, thresholds = roc_curve(y_test, y_prob_tuned)

plt.figure(figsize=(8, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC 曲線 (AUC = {auc_tuned:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('假陽性率 (False Positive Rate, FPR)')
plt.ylabel('真陽性率 (True Positive Rate, TPR) / 召回率')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

# 繪製特徵
importances = best_rf.feature_importances_
feature_importances = pd.Series(importances, index=X_train.columns)
sorted_importances = feature_importances.sort_values(ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(
x=sorted_importances[:10].values,
y=sorted_importances[:10].index,
palette="Reds_d"
)
plt.title("最佳模型特徵重要性排名")
plt.xlabel("重要性分數")
plt.ylabel("特徵")
plt.tight_layout()
plt.show()

# --- 視覺化最佳模型的前 10 名特徵重要性 ---
importances = best_rf.feature_importances_
feature_names = X_train.columns
feature_importance_df = pd.DataFrame({'特徵': feature_names, '重要性': importances})

top_10_features = feature_importance_df.sort_values(by='重要性', ascending=False).head(10)

plt.figure(figsize=(10, 6))

try:
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False 
except:
    pass

sns.barplot(
    x='重要性', 
    y='特徵', 
    data=top_10_features, 
    palette="viridis" 
)

plt.title(f"Grid Search 最佳模型 - 前 10 名特徵重要性\n(最佳參數: {grid_search.best_params_})", fontsize=14)
plt.xlabel("重要性分數 (Gini Importance)", fontsize=12)
plt.ylabel("特徵名稱", fontsize=12)
plt.tight_layout()
plt.show()