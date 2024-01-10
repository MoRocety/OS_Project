import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Load the dataset
data = pd.read_csv("informed_dataset_refined.csv")

# Identify categorical columns
categorical_cols = data.select_dtypes(include=['object']).columns.tolist()

# If there are categorical columns, perform encoding
if categorical_cols:
    # One-Hot Encoding for linear models
    encoder = OneHotEncoder(drop='first', sparse_output=False)
    data_encoded = encoder.fit_transform(data[categorical_cols])
    
    # Get the names of the one-hot encoded columns
    encoded_column_names = encoder.get_feature_names_out(categorical_cols)
    
    # Create a DataFrame with one-hot encoded columns and original columns
    data_encoded = pd.concat([data.drop(categorical_cols, axis=1),
                              pd.DataFrame(data_encoded, columns=encoded_column_names)], axis=1)

    # Convert column names to strings
    data_encoded.columns = data_encoded.columns.astype(str)
else:
    data_encoded = data.copy()

# Extract features and target variable
X = data_encoded.drop("RunTime", axis=1)
y = data_encoded["RunTime"]

# Print the original column names (features)
original_column_names = X.columns
print("Original Column names:", original_column_names)

# Replace the numerical column names with the original categorical variable names
X.columns = original_column_names

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create DataFrames for the training and testing sets
train_data = pd.concat([X_train, y_train], axis=1)
test_data = pd.concat([X_test, y_test], axis=1)

# Save the training and testing sets to separate CSV files
train_data.to_csv('train_data_fossil.csv', index=False)
test_data.to_csv('test_data_fossil.csv', index=False)

# Create a "models" directory if it doesn't exist
models_directory = 'models'
os.makedirs(models_directory, exist_ok=True)

# k-Nearest Neighbors Regression
knn_reg = KNeighborsRegressor(n_neighbors=3)
knn_reg.fit(X_train, y_train)
knn_pred = knn_reg.predict(X_test)
knn_mse = mean_squared_error(y_test, knn_pred)
knn_r2 = r2_score(y_test, knn_pred)
knn_corr = np.corrcoef(y_test, knn_pred)[0, 1]

print("k-Nearest Neighbors Regression MSE:", knn_mse)
print("k-Nearest Neighbors Regression R^2:", knn_r2)
print("k-Nearest Neighbors Regression Correlation Coefficient:", knn_corr)

print()

# Decision Tree Regression
tree_reg = DecisionTreeRegressor()
tree_reg.fit(X_train, y_train)
tree_pred = tree_reg.predict(X_test)
tree_mse = mean_squared_error(y_test, tree_pred)
tree_r2 = r2_score(y_test, tree_pred)
tree_corr = np.corrcoef(y_test, tree_pred)[0, 1]

print("Decision Tree Regression MSE:", tree_mse)
print("Decision Tree Regression R^2:", tree_r2)
print("Decision Tree Regression Correlation Coefficient:", tree_corr)
print()

# Save the Decision Tree Regression model
joblib.dump(tree_reg, 'models/informed_decision_tree_regression_model.joblib')

# Random Forest Regression
random_forest_reg = RandomForestRegressor(n_estimators=100, random_state=42)
random_forest_reg.fit(X_train, y_train)
random_forest_pred = random_forest_reg.predict(X_test)
random_forest_mse = mean_squared_error(y_test, random_forest_pred)
random_forest_r2 = r2_score(y_test, random_forest_pred)
random_forest_corr = np.corrcoef(y_test, random_forest_pred)[0, 1]

print("Random Forest Regression MSE:", random_forest_mse)
print("Random Forest Regression R^2:", random_forest_r2)
print("Random Forest Regression Correlation Coefficient:", random_forest_corr)
print()

# Save the Random Forest Regression model
joblib.dump(random_forest_reg, 'models/informed_random_forest_regression_model.joblib')
