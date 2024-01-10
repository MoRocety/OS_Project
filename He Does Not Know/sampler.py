import pandas as pd
import random

def select_random_percentage(input_csv, output_csv, percentage=10):
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(input_csv)

    # Calculate the number of rows to select based on the specified percentage
    num_rows_to_select = int(len(df) * (percentage / 100.0))

    # Randomly select rows
    selected_rows = random.sample(range(len(df)), num_rows_to_select)

    # Create a new DataFrame with the selected rows
    selected_df = df.iloc[selected_rows]

    # Save the selected data to a new CSV file with the header
    selected_df.to_csv(output_csv, index=False)

if __name__ == "__main__":
    # Specify the input CSV file and the output CSV file
    input_csv_file = "hedoesnotknow_file.csv"
    output_csv_file = "hedoesnotknow_file.csv"

    # Call the function to select 10% of the data and save it to a new CSV file
    select_random_percentage(input_csv_file, output_csv_file, percentage=10)

    print(f"Random 10% of data saved to {output_csv_file}")
