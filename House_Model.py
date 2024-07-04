import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Sample dataset (for demonstration purposes)
data = {
    'OverallQual': [7, 6, 7, 7, 8, 5, 5, 8, 7, 7],
    'GrLivArea': [1710, 1262, 1786, 1717, 2198, 1362, 1694, 2090, 1774, 1077],
    'GarageCars': [2, 2, 2, 3, 3, 2, 2, 3, 2, 2],
    'GarageArea': [548, 460, 608, 642, 836, 480, 636, 736, 520, 480],
    'TotalBsmtSF': [856, 1262, 920, 756, 1145, 796, 1686, 860, 920, 996],
    'SalePrice': [208500, 181500, 223500, 140000, 250000, 143000, 307000, 200000, 129900, 118000]
}
df = pd.DataFrame(data)

# Features and target variable
X = df.drop('SalePrice', axis=1)
y = df['SalePrice']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the column transformer for preprocessing
numeric_features = ['OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea', 'TotalBsmtSF']
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features)
    ])

# Define the model pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print(f'MAE: {mae}')
print(f'MSE: {mse}')
