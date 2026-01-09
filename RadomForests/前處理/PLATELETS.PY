import pandas as pd     
import seaborn as sns    
import matplotlib.pyplot as plt

df = pd.read_csv("1.前處理報告數據.csv")

plt.figure(figsize=(8,5))
sns.boxplot(x=df["platelets"], color='lightcoral')
plt.title("血小板")
plt.xlabel("platelets")
plt.show()

plt.figure(figsize=(8,5))
plt.hist(df["platelets"], bins=20, color='skyblue', edgecolor='black')
plt.title("血小板")
plt.xlabel("platelets")
plt.ylabel("人數")
plt.show()
