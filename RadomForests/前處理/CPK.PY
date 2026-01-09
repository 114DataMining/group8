import pandas as pd      
import seaborn as sns    
import matplotlib.pyplot as plt

df = pd.read_csv("1.前處理報告數據.csv")

plt.figure(figsize=(8,5))
sns.boxplot(x=df["creatininephosphokinase"], color='lightcoral')
plt.title("肌酸酐磷酸激酶")
plt.xlabel("creatinine phosphokinase")
plt.show()

plt.figure(figsize=(8,5))
plt.hist(df["creatininephosphokinase"], bins=20, color='skyblue', edgecolor='black')
plt.title("肌酸酐磷酸激酶")
plt.xlabel("creatinine phosphokinase")
plt.ylabel("人數")
plt.show()
