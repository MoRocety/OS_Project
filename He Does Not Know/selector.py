import pandas as pd

df = pd.read_csv('hedoesnotknow_file.csv')

columns_to_remove = [          
            'OrigSiteID',
              ]
df = df.drop(columns=columns_to_remove)

df.to_csv('hedoesnotknow_file.csv', index=False)

print(f"Columns {columns_to_remove} removed. Modified data saved.")
