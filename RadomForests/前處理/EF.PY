import pandas as pd      
import seaborn as sns    
import matplotlib.pyplot as plt

df = pd.read_csv("1.前處理報告數據.csv")

plt.figure(figsize=(8,5))
sns.boxplot(x=df["ejectionfraction"], color='lightcoral')
plt.title("射血分數")
plt.xlabel("ejection fraction")
plt.show()

plt.figure(figsize=(8,5))
plt.hist(df["ejectionfraction"], bins=20, color='skyblue', edgecolor='black')
plt.title("射血分數")
plt.xlabel("ejection fraction")
plt.ylabel("人數")
plt.show()