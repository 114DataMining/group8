import pandas as pd      
import seaborn as sns    
import matplotlib.pyplot as plt

df = pd.read_csv("1.前處理報告數據.csv")

plt.figure(figsize=(8,5))
sns.boxplot(x=df["serumcreatinine"], color='lightcoral')
plt.title("血清肌酸酐盒形圖")
plt.xlabel("serum creatinine")
plt.show()

plt.figure(figsize=(8,5))
plt.hist(df["serumcreatinine"], bins=20, color='skyblue', edgecolor='black')
plt.title("血清肌酸酐直方圖")
plt.xlabel("serum creatinine")
plt.ylabel("人數")
plt.show()
