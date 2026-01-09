import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


try:
    df_test = pd.read_csv('heartattack_test_20_1202.csv') 
    df_train = pd.read_csv('heartattack_train_80_1202.csv') 
    print("訓練集和測試集數據匯入成功!")
except FileNotFoundError:
    print("錯誤：找不到。")
    df_test =  pd.DataFrame()
    df_train = pd.DataFrame()


feature_cols = [
    'Zejectionfraction', 'Zserumsodium' , 'Zcreatininephosphokinase',
    'age_new', 'Zplatelets',  'diabetes', 'highbp', 'sex', 'smoking','anaemia'
]
target_col = 'Zserumcreatinine' 


if not df_train.empty and not df_test.empty:
    X_train = df_train[feature_cols]
    y_train = df_train[target_col]
    
    X_test = df_test[feature_cols]
    y_test = df_test[target_col]
    
    print(f"訓練集大小: {X_train.shape[0]} 筆")
    print(f"測試集大小: {X_test.shape[0]} 筆")

    print("\n開始訓練")
    rf_regressor = RandomForestRegressor(
    n_estimators=200,    
    random_state=42,    
    max_depth=None     
    )
    rf_regressor.fit(X_train, y_train)
    print("隨機森林模型訓練完成!")

    y_pred = rf_regressor.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print("--- 隨機森林模型性能 (測試集) ---")
    print(f"R-squared (R^2): {r2:.4f}")
    print(f"MAE (平均絕對誤差): {mae:.4f}")
    print(f"RMSE (均方根誤差): {rmse:.4f}")

    importances = rf_regressor.feature_importances_
    feature_importances = pd.Series(importances, index=X_train.columns)
    sorted_importances = feature_importances.sort_values(ascending=False)

    print("\n--- 特徵重要性排名 ---")
    print(sorted_importances)
    
    sns.set_style("whitegrid")
    plt.figure(figsize=(10, 6))
    top_n = 10
    n_features =min(top_n, len(sorted_importances))

    try:
        plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'DejaVu Sans'] 
        plt.rcParams['axes.unicode_minus'] = False 
    except:
        pass
    
    sns.barplot(
    x=sorted_importances[:n_features].values, 
    y=sorted_importances[:n_features].index, 
    palette="Blues_d"
    )
    plt.title(f"Random Forest Feature Importance (Top {n_features})")
    plt.xlabel("Importance Score")
    plt.ylabel("Features")
    plt.tight_layout()
    plt.show() 
else:
    print("\n失敗")