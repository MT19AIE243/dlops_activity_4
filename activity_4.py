import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Sample dataset
data = {
    'square_feet': [1000, 1500, 2000, 2500, 3000],
    'num_bedrooms': [2, 3, 3, 4, 4],
    'location': ['A', 'B', 'C', 'A', 'B'],
    'price': [150000, 200000, 250000, 300000, 350000]
}
df = pd.DataFrame(data)

# Encode categorical variables
df = pd.get_dummies(df, columns=['location'])

# Split dataset into training and testing sets
X = df.drop('price', axis=1)
y = df['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Regressor
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
