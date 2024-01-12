import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import LabelEncoder
import random

# Load your data
df = pd.read_csv('../Data_Cleaning/informed_dataset.csv')

# Convert categorical variables to numerical
le = LabelEncoder()
categorical_features = df.select_dtypes(include=['object']).columns
for feature in categorical_features:
    df[feature] = le.fit_transform(df[feature])

# Define your features and target variable
X = df.drop('RunTime', axis=1)
y = df['RunTime']

# Feature selection using RandomForestRegressor to get feature importance scores
model = RandomForestRegressor()
model.fit(X, y)

# Feature selection using SelectKBest with f_regression
k_best = 3
skb = SelectKBest(f_regression, k=k_best)
X_selected = skb.fit_transform(X, y)

# Get the indices of the selected features
selected_feature_indices = skb.get_support(indices=True)

# Include 'SubmitTime' column in the selected features
selected_feature_indices = list(set(selected_feature_indices) | {X.columns.get_loc('SubmitTime')})

# Create a DataFrame with the selected features and the target variable
selected_df = pd.DataFrame(X.iloc[:, selected_feature_indices])
selected_df['RunTime'] = y

# Print the selected features
print("\nTop selected features with the target variable:")
print(selected_df.head())

# Take a random 10% sample
percentage = 10
num_rows_to_select = int(len(selected_df) * (percentage / 100.0))
selected_rows = random.sample(range(len(selected_df)), num_rows_to_select)
selected_df_sampled = selected_df.iloc[selected_rows]

# Save the selected data to a new CSV file with the header
output_csv_file = "informed_dataset_refined.csv"
selected_df_sampled.to_csv(output_csv_file, index=False)

print(f"\nTop {k_best} features selected, random 10% of data saved to {output_csv_file}")
