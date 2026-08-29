import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv("dataset/electricity_cost_dataset.csv")
print(df.head())
print(df.shape)
print(df.columns)
print(df.isnull().sum())
print(df.describe())
df = pd.get_dummies(df, columns=["structure type"], drop_first=True)
X = df.drop("electricity cost", axis=1)
y = df["electricity cost"]
print(X.head())
print(y.head())
plt.scatter(df["site area"], df["electricity cost"])
plt.xlabel("Site Area")
plt.ylabel("Electricity Cost")
plt.title("Site Area vs Electricity Cost")
plt.show()