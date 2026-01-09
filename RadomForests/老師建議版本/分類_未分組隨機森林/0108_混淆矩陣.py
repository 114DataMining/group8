import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score
import matplotlib.pyplot as plt

train_file = 'heartattack_train_80_CLF.csv'
test_file = 'heartattack_test_20_CLF.csv'


feature_cols = [
    'Zejectionfraction', 'Zplatelets', 'Zserumsodium',  
    'Zserumcreatinine', 'Zcreatininephosphokinase', 'age_new',
    'anaemia', 'diabetes', 'highbp', 'sex', 'smoking'
]


BEST_PARAMS = {
    'class_weight': 'balanced',
    'max_depth': 4,
    'max_features': 'sqrt',
    'min_samples_leaf': 3,
    'n_estimators': 201,
    'random_state': 42
}


try:
    df_train = pd.read_csv(train_file)
    df_test = pd.read_csv(test_file)
   
    X_train = df_train[feature_cols]
    y_train = df_train['death']
    X_test = df_test[feature_cols]
    y_test = df_test['death']

   
    rf = RandomForestClassifier(**BEST_PARAMS)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    y_prob = rf.predict_proba(X_test)[:, 1]

   
    print("\n" + "="*40)
    print("      最優模型20% ")
    print("="*40)
    print(f"AUC: {roc_auc_score(y_test, y_prob):.4f}")
    print("\n詳細分類指標：")
    print(classification_report(y_test, y_pred, target_names=['存活', '死亡']))

       
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'Arial']
    plt.rcParams['axes.unicode_minus'] = False

    fig, ax = plt.subplots(figsize=(7, 6))
    cm = confusion_matrix(y_test, y_pred)
   
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['預測存活', '預測死亡'])
    disp.plot(cmap='YlGnBu', ax=ax, values_format='d')
   
    plt.title(' 混淆矩陣 (Confusion Matrix)')
    plt.grid(False)
    plt.show()

except Exception as e:
    print(f"發生錯誤：{e}")
