import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, f1_score, recall_score, roc_curve
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

BEST_PARAMS = {
    'class_weight': 'balanced',
    'max_depth': 4,
    'max_features': 'sqrt',
    'min_samples_leaf': 3,
    'n_estimators': 200,
    'random_state': 42 
}

try:
    df_full = pd.read_csv('1215dataset_new.csv') 
    print("已成功匯入分組特徵的數據。")
except FileNotFoundError:
    print("錯誤：找不到檔案。")
    exit()

feature_cols = [
    'G_ejectionfraction', 
    'G_platelets', 
    'G_serumsodium',  
    'G_serumcreatinine', 
    'G_creatininephosphokinase', 
    'G_age', 
    'anaemia', 
    'diabetes', 
    'highbp', 
    'sex', 
    'smoking' 
]
target_col = 'death' 

missing_cols = [col for col in feature_cols if col not in df_full.columns]
if missing_cols:
    print(f"錯誤：檔案中缺少欄位")
    exit()

df_full.dropna(subset=feature_cols + [target_col], inplace=True)

X_full = df_full[feature_cols]
y_full = df_full[target_col]

X_train, X_test, y_train, y_test = train_test_split(
    X_full, 
    y_full, 
    test_size=0.2, 
    random_state=42,
    stratify=y_full 
)
print(f"數據已重新切割：訓練集樣本數={len(X_train)}，測試集樣本數={len(X_test)}")

# 訓練與評估
print("使用分組特徵與優化參數訓練隨機森林分類器")
final_rf = RandomForestClassifier(**BEST_PARAMS)
final_rf.fit(X_train, y_train)

# 進行預測
y_pred_new = final_rf.predict(X_test)
y_prob_new = final_rf.predict_proba(X_test)[:, 1]

# 計算性能指標
f1_tuned = f1_score(y_test, y_pred_new, pos_label=1)
recall_tuned = recall_score(y_test, y_pred_new, pos_label=1)
report_tuned = classification_report(y_test, y_pred_new)
auc_tuned = roc_auc_score(y_test, y_prob_new)
accuracy_tuned = final_rf.score(X_test, y_test)

print("\n--- 模型表現 (使用分組特徵) ---")
print(f"整體準確度 (Accuracy): {accuracy_tuned:.4f}")
print(f"調優後的 F1-Score (Death=1): {f1_tuned:.4f} (平衡指標)")
print(f"調優後的召回率 (Recall for Death=1): {recall_tuned:.4f}")
print(f"調優後的 AUC: {auc_tuned:.4f}")
print("\n詳細分類報告 (Classification Report):")
print(report_tuned)

try:
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False 
except NameError:
    pass 

sns.set_style("whitegrid")
fpr, tpr, thresholds = roc_curve(y_test, y_prob_new)

plt.figure(figsize=(8, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC 曲線 (AUC = {auc_tuned:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('假陽性率 (False Positive Rate, FPR)')
plt.ylabel('真陽性率 (True Positive Rate, TPR) / 召回率')
plt.title('ROC分組 Curve')
plt.legend(loc="lower right")
plt.show()

try: 
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'DejaVu San' ]
    plt.rcParams['axes.unicode_minus'] = False
except NameError:
    pass  
importances = final_rf.feature_importances_
feature_importances = pd.Series ( importances, index=X_train.columns) 
sorted_importances = feature_importances.sort_values(ascending=False)

plt.figure(figsize=(10, 6)) 
sns.barplot( 
    x=sorted_importances[:10].values, 
    y=sorted_importances[:10].index,
    palette="Reds_d" 
) 
plt.title("随機森林特徵重要性排名" )
plt.xlabel("正要性特徵") 
plt.ylabel("特徵") 
plt.tight_layout() 
plt.show()

