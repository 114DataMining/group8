import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score


BEST_PARAMS = {
    'class_weight': 'balanced',
    'max_depth': 4,
    'max_features': 'sqrt',
    'min_samples_leaf': 3,
    'n_estimators': 200,
    'random_state': 42
}

feature_cols = [
    'Zejectionfraction', 'Zplatelets', 'Zserumsodium',  
    'Zserumcreatinine', 'Zcreatininephosphokinase', 'age_new',
    'anaemia', 'diabetes', 'highbp', 'sex', 'smoking'
]


try:
    df_train = pd.read_csv('heartattack_train_80_CLF.csv')
    df_test = pd.read_csv('heartattack_test_20_CLF.csv')
except:
    print("檔案讀取失敗。")
    exit()


rf = RandomForestClassifier(**BEST_PARAMS)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_auc = cross_val_score(rf, df_train[feature_cols], df_train['death'], cv=cv, scoring='roc_auc').mean()


rf.fit(df_train[feature_cols], df_train['death'])
test_auc = roc_auc_score(df_test['death'], rf.predict_proba(df_test[feature_cols])[:, 1])

print("\n" + "="*40)
print(f"五折驗證平均 AUC (80% 訓練集): {cv_auc:.4f}")
print(f"單次測試 AUC (20% 測試集): {test_auc:.4f}")
print("="*40)

if test_auc < cv_auc - 0.1:
    print("警告：測試集分數掉太多了。")
else:
    print("狀態：分數落差在合理範圍內。")