import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np


ORIGINAL_FILE_NAME = 'heartattack_original_297_1.csv' 
print(f"開始處理原始數據檔案: {ORIGINAL_FILE_NAME}")

try:
    df_original = pd.read_csv(ORIGINAL_FILE_NAME) 
    
    if 'death' not in df_original.columns:
        print("原始數據中找不到 'death' 欄位。")
        exit()
        
except FileNotFoundError:
    print(f"錯誤：找不到原始數據檔案 {ORIGINAL_FILE_NAME}。")
    exit()


CONTINUOUS_COLS_TO_STANDARDIZE = [
    'creatininephosphokinase', 
    'ejectionfraction',       
    'serumsodium',            
    'serumcreatinine',        
    'platelets'
]

# 劃分訓練集和測試集 (80/20) 
df_train, df_test = train_test_split(
    df_original, 
    test_size=0.2, 
    random_state=42, 
    stratify=df_original['death'] 
)

print(f"數據劃分完成：訓練集 {df_train.shape[0]} 筆，測試集 {df_test.shape[0]} 筆 ")


for col in CONTINUOUS_COLS_TO_STANDARDIZE:
    
    mean = df_train[col].mean()
    std = df_train[col].std()
    new_z_col_name = f'Z{col}'
    

    df_train[new_z_col_name] = (df_train[col] - mean) / std
    df_test[new_z_col_name] = (df_test[col] - mean) / std

print("\nZ-Score 標準化完成。")


CATEGORICAL_COLS = ['age_new', 'anaemia', 'diabetes', 'highbp', 'sex', 'smoking']
TARGET_COL = 'death' 


all_z_cols = [f'Z{col}' for col in CONTINUOUS_COLS_TO_STANDARDIZE]
final_cols = all_z_cols + CATEGORICAL_COLS + [TARGET_COL]

df_train_final = df_train[final_cols].copy()
df_test_final = df_test[final_cols].copy()


df_train_final.to_csv('heartattack_train_80_CLF.csv', index=False)
df_test_final.to_csv('heartattack_test_20_CLF.csv', index=False)

print("\n--- 數據輸出完成 ---")