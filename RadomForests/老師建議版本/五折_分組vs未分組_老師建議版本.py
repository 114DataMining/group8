import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import recall_score, f1_score, roc_auc_score

BEST_PARAMS = {
    'class_weight': 'balanced',
    'max_depth': 4,
    'max_features': 'sqrt',
    'min_samples_leaf': 3,
    'n_estimators': 201,
    'random_state': 42 
}

try:
   
    df = pd.read_csv('五折_80_1223.csv')
    print("數據載入成功")
except:
    print("找不到資料檔")
    exit()

target = 'death'


original_features = [
    'Zejectionfraction', 'Zplatelets', 'Zserumsodium',  
    'Zserumcreatinine', 'Zcreatininephosphokinase', 'age_new', 
    'anaemia', 'diabetes', 'highbp', 'sex', 'smoking' 
]


grouped_features = [
    'G_ejectionfraction', 'G_platelets', 'G_serumsodium',  
    'G_serumcreatinine', 'G_creatininephosphokinase', 'G_age', 
    'anaemia', 'diabetes', 'highbp', 'sex', 'smoking' 
]


skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def evaluate_model_cv(features, name):
    
    missing = [f for f in features if f not in df.columns]
    if missing:
        return {'模型描述': name, '錯誤': f"缺失欄位: {missing}"}
    
    X = df[features]
    y = df[target]
    rf = RandomForestClassifier(**BEST_PARAMS)
    
    auc_scores = cross_val_score(rf, X, y, cv=skf, scoring='roc_auc')
    f1_scores = cross_val_score(rf, X, y, cv=skf, scoring='f1')
    
    return {
        '模型描述': name,
        '平均 AUC': f"{auc_scores.mean():.4f} (±{auc_scores.std()*2:.4f})",
        '平均 F1-Score': f"{f1_scores.mean():.4f}",
        '特徵數量': len(features)
    }


results = []
results.append(evaluate_model_cv(original_features, "原始連續變數模型 (推薦)"))
results.append(evaluate_model_cv(grouped_features, "類別分組模型 (觀察損失)"))

# 輸出結果
results_df = pd.DataFrame(results)
print("\n" + "="*70)
print("             五折交叉驗證：連續 vs 分組 性能對照表")
print("="*70)
print(results_df.to_string(index=False))
print("="*70)