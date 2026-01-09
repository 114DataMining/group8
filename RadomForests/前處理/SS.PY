import pandas as pd      
import seaborn as sns    
import matplotlib.pyplot as plt

df = pd.read_csv("1.前處理報告數據.csv")

plt.figure(figsize=(8,5))
sns.boxplot(x=df["serumsodium"], color='lightcoral')
plt.title("血清鈉")
plt.xlabel("serum sodium")
plt.show()

plt.figure(figsize=(8,5))
plt.hist(df["serumsodium"], bins=20, color='skyblue', edgecolor='black')
plt.title("血清鈉")
plt.xlabel("serumsodium")
plt.ylabel("人數")
plt.show()
