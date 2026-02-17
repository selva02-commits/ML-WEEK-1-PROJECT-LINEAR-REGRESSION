import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load dataset
df = pd.read_csv('earthquakes.csv')

# Use the correct column names
X = df[['impact.magnitude']]   # predictor column
y = df['location.depth']       # target column

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Equation
m = model.coef_[0]
c = model.intercept_
print(f"Linear Regression Equation: Depth = {m:.2f} × Magnitude + {c:.2f}")

# Prediction
user_value = float(input("Enter magnitude: "))
user_df = pd.DataFrame({'impact.magnitude': [user_value]})
predicted_y = model.predict(user_df)
print(f"Predicted depth: {predicted_y[0]:.2f}")
