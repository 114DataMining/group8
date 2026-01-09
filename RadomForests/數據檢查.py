import pandas as pd
import numpy as np

try:
    df_train = pd.read_csv('heartattack_train_80_CLF.csv') 
    df_test= pd.read_csv('heartattack_test_20_CLF.csv') 
except:
    print ("讀取失敗。" )
    exit()

train_raw=df_train.drop(columns=['death']).values 
test_raw= df_test.drop(columns=['death']).values

overlaps =0
for row in test_raw:
    if any(np.equal(train_raw, row).all(1)): 
        overlaps +=1

print("="*50)
print("【數據洩漏稽核報告】")
print("="*50)
if overlaps ==0:
    print (f"安全： 測試集中的 {len(df_test)} 筆數據在訓練集中皆未出現。")
else:
    print (f" 發現overlaps 筆重疊数據")

train_rate = df_train['death'].mean() 
test_rate = df_test['death'].mean( )
print(f" \n訓練集死亡率： [train_rate:.2%]") 
print(f"測試集先亡率： ltest_rate: .2%]")


print (f"1. 測試集樣本僅 {len(df_test)}筆AUC") 
print(f"2. 若樣本無重疊 ,則 AUC 0.86 代表模型具有的「外推能力") 
print("="*50)
