import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve

# 匯入數據
try:
    df_test = pd.read_csv('heartattack_test_20_CLF.csv') 
    df_train = pd.read_csv('heartattack_train_80_CLF.csv') 
    print("訓練集和測試集 數據匯入成功!")
except FileNotFoundError:
    print("錯誤：找不到 CLF 檔案。")
    df_train = pd.DataFrame()
    df_test = pd.DataFrame()
    
feature_cols = [
    'Zejectionfraction', 'Zplatelets', 'Zserumsodium',  
    'Zserumcreatinine', 'Zcreatininephosphokinase', 'age_new', 
    'anaemia', 'diabetes', 'highbp', 'sex', 'smoking' 
]
target_col = 'death' 

# 模型訓練與評估 
if not df_train.empty and not df_test.empty:
    X_train = df_train[feature_cols]
    y_train = df_train[target_col]

    X_test = df_test[feature_cols]
    y_test = df_test[target_col]

    print(f"\n訓練集大小: {X_train.shape[0]} 筆")
    print(f"測試集大小: {X_test.shape[0]} 筆")

    
    print("\n開始訓練隨機森林分類模型")
    rf_classifier = RandomForestClassifier(
        n_estimators=200,    
        random_state=42,     
        max_depth=None       
    )

    rf_classifier.fit(X_train, y_train)
    print("隨機森林分類模型訓練完成!")

    #使用測試集進行預測與評估
    y_pred = rf_classifier.predict(X_test)
    y_prob = rf_classifier.predict_proba(X_test)[:, 1] 

    
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    auc_score = roc_auc_score(y_test, y_prob)

    print("\n--- 隨機森林模型(分類) ---")
    print(f"整體準確度 (Accuracy): {accuracy:.4f}") 
    print(f"AUC (曲線下面積): {auc_score:.4f}")
    print("\n詳細分類報告 (Classification Report):")
    print(report)

    # 視覺化特徵
    importances = rf_classifier.feature_importances_
    feature_importances = pd.Series(importances, index=X_train.columns)
    sorted_importances = feature_importances.sort_values(ascending=False)

    print("\n--- 特徵重要性排名 ---")
    print(sorted_importances)

    # 步驟 4：視覺化 ROC 曲線 
    sns.set_style("whitegrid")
    
    # 視覺化 ROC 曲線
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_score:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    
    
    plt.figure(figsize=(10, 6))
    
    try:
        plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False 
    except:
        pass 

    sns.barplot(
        x=sorted_importances[:10].values, 
        y=sorted_importances[:10].index, 
        palette="Reds_d"
    )
    plt.title("隨機森林特徵重要性 (預測死亡率)")
    plt.xlabel("重要性分數")
    plt.ylabel("特徵")
    plt.tight_layout() 
    plt.show() 
else:
    print("\n無法進行模型訓練。")