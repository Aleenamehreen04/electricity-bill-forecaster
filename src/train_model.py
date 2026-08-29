import pandas as pd
from sklearn.model_selection import train_test_split

# Load data
df = pd.read_csv("dataset/electricity_cost_dataset.csv")

# Preprocessing
df = pd.get_dummies(df, columns=["structure type"], drop_first=True)
X = df.drop("electricity cost", axis=1)
y = df["electricity cost"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(X_train.shape, X_test.shape)
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)
print(predictions[:5])
print(y_test[:5].values)