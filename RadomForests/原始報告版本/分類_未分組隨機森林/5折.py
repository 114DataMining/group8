import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc, roc_auc_score


BEST_PARAMS = {
    'class_weight': 'balanced',
    'max_depth': 4,
    'max_features': 'sqrt',
    'min_samples_leaf': 3,
    'n_estimators': 200,
    'random_state': 42 
}

try:
    df = pd.read_csv('五折_80_1223.csv')
    print("數據載入成功")
except:
    print("找不到資料檔。")
    exit()


feature_cols = [
    'Zejectionfraction', 'Zplatelets', 'Zserumsodium',  
    'Zserumcreatinine', 'Zcreatininephosphokinase', 'age_new', 
    'anaemia', 'diabetes', 'highbp', 'sex', 'smoking' 
]
target_col = 'death'

X = df[feature_cols]
y = df[target_col]


cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
classifier = RandomForestClassifier(**BEST_PARAMS)

tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

plt.figure(figsize=(8, 8))


try:
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'Arial']
    plt.rcParams['axes.unicode_minus'] = False
except:
    pass

for i, (train, test) in enumerate(cv.split(X, y)):
    
    classifier.fit(X.iloc[train], y.iloc[train])
    
    
    viz = roc_curve(y.iloc[test], classifier.predict_proba(X.iloc[test])[:, 1])
    interp_tpr = np.interp(mean_fpr, viz[0], viz[1])
    interp_tpr[0] = 0.0
    tprs.append(interp_tpr)
    
    roc_auc = auc(viz[0], viz[1])
    aucs.append(roc_auc)
    
    
    plt.plot(viz[0], viz[1], lw=1, alpha=0.3, label=f'Fold {i+1} (AUC = {roc_auc:.2f})')


plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Random Guess', alpha=.8)


mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)


plt.plot(mean_fpr, mean_tpr, color='b',
         label=f'Mean ROC (AUC = {mean_auc:.2f} $\pm$ {std_auc:.2f})',
         lw=2, alpha=.8)


std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                 label='$\pm$ 1 std. dev.')

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate (假陽性率)')
plt.ylabel('True Positive Rate (真陽性率)')
plt.title('5-Fold Cross Validation - Best Model ROC')
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.show()

print(f"\n五折驗證完成！平均 AUC 為: {mean_auc:.4f}")
print(f"各折 AUC 標偏: {std_auc:.4f}")