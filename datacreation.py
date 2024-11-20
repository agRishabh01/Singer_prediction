import pandas as pd
import numpy as np
import os

# Folder path containing the CSV files
folder_path = "new_data"  # Replace with the path to your folder

# List to store stats for each file
stats_list = []

# Loop through each file in the folder
for filename in os.listdir(folder_path):
    if filename.endswith(".csv"):
        file_path = os.path.join(folder_path, filename)
        data = pd.read_csv(file_path, header=None)
        
        # Dictionary to hold the current file's stats
        file_stats = {'filename': filename}
        
        for i in range(len(data)):
            # Convert row data to numeric, forcing non-numeric values to NaN
            row_data = pd.to_numeric(data.iloc[i], errors='coerce')
            
            # Calculate statistical features for the current row, ignoring NaN values
            file_stats[f'mean{i+1}'] = row_data.mean()
            file_stats[f'median{i+1}'] = row_data.median()
            file_stats[f'variance{i+1}'] = row_data.var()
            file_stats[f'std_dev{i}'] = row_data.std()
            file_stats[f'skewness{i}'] = row_data.skew()
            # file_stats[f'kurtosis{i}'] = row_data.kurtosis()
            
        
        # Append stats to list
        stats_list.append(file_stats)

# Create DataFrame and save to Excel
stats_df = pd.DataFrame(stats_list)
stats_df.to_excel("new_dataset.xlsx", index=False)

print("Statistical summary saved to 'new_dataset.xlsx'")
