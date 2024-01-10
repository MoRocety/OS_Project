import pandas as pd

# Load the CSV file
input_file = 'full_cleaned_dataset.csv'

output_files = {"uninformed_dataset.csv": ["SubmitTime", "UserID", "Used Memory", "OrigSiteID", "GroupID", "RunTime"],
                "informed_dataset.csv": ["SubmitTime", "UserID", "Used Memory", "OrigSiteID", "GroupID", "AverageCPUTimeUsed", "RunTime"]}

for output_file in output_files:
    # Read the CSV file
    df = pd.read_csv(input_file)

    # Retain only the desired columns
    df_retained = df[output_files[output_file]]

    # Save the result to a new CSV file
    df_retained.to_csv(output_file, index=False)

    print(f"Columns retained: {output_files[output_file]}")
    print(f"Output saved to: {output_file}")
