import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import LabelEncoder
import random

# Load your data
df = pd.read_csv('../Data Cleaning/uninformed_dataset.csv')

# Convert categorical variables to numerical
le = LabelEncoder()
categorical_features = df.select_dtypes(include=['object']).columns
for feature in categorical_features:
    df[feature] = le.fit_transform(df[feature])

# Define your features and target variable
X = df.drop('RunTime', axis=1)
y = df['RunTime']

# Feature selection using RandomForestRegressor
model = RandomForestRegressor()
sfm = SelectFromModel(model, threshold=0.05)
sfm.fit(X, y)

# Retain only the top 3 features
selected_features = X.columns[sfm.get_support()]
X_selected = X[selected_features]

# Add the target variable back to the selected features
X_selected['RunTime'] = y

# Print the selected features
print("Top 3 selected features with the target variable:")
print(X_selected.head())

# Take a random 10% sample
percentage = 10
num_rows_to_select = int(len(X_selected) * (percentage / 100.0))
selected_rows = random.sample(range(len(X_selected)), num_rows_to_select)
selected_df = X_selected.iloc[selected_rows]

# Save the selected data to a new CSV file with the header
output_csv_file = "uninformed_dataset_refined.csv"
selected_df.to_csv(output_csv_file, index=False)

print(f"Top 3 features selected, random 10% of data saved to {output_csv_file}")
