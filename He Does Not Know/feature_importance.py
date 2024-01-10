import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# Load your data
df = pd.read_csv('hedoesnotknow_file.csv')

# Convert categorical variables to numerical
le = LabelEncoder()
categorical_features = df.select_dtypes(include=['object']).columns
for feature in categorical_features:
    df[feature] = le.fit_transform(df[feature])

# Define your features and target variable
X = df.drop('RunTime', axis=1)
y = df['RunTime']

# Fit the model
model = RandomForestRegressor()
model.fit(X, y)

# Get feature importances
importances = model.feature_importances_

# Create a DataFrame for easy viewing
feature_importances = pd.DataFrame({'Feature': X.columns, 'Importance': importances})

# Print the feature importances
print(feature_importances.sort_values(by='Importance', ascending=False))

