import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import os
# Read data from Excel file
file_path = 'mj2.xlsx'
data = pd.read_excel(file_path)

# Separate features and target variable
X = data.drop(columns=['filename'])
X.dropna()
y = data['filename']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)



# Algorithms
algorithms = {
        'Logistic Regression': LogisticRegression(),
        'SVC (Linear)': SVC(kernel='linear',probability=True),
        'SVC (RBF)': SVC(kernel='rbf',probability=True),
        'Random Forest (msl=1)': RandomForestClassifier(min_samples_leaf=1),
        'Random Forest (msl=3)': RandomForestClassifier(min_samples_leaf=3),
        'Random Forest (msl=5)': RandomForestClassifier(min_samples_leaf=6)}

# Metric tables
metric_table_train = pd.DataFrame()
metric_table_test = pd.DataFrame()



# Run the algorithms, create metrics, and plots
for algorithm_name, algorithm in algorithms.items():
    algorithm.fit(X_train, y_train)

    # Train predictions
    y_train_pred = algorithm.predict(X_train)
    # Test predictions
    y_test_pred = algorithm.predict(X_test)

    # Train metrics
   
    
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    
    train_precision_perClass = precision_score(y_train, y_train_pred, average = None)
    test_precision_perClass = precision_score(y_test, y_test_pred, average = None)

    train_precision_average = precision_score(y_train, y_train_pred, average= 'macro')
    test_precision_average = precision_score(y_test, y_test_pred, average='macro')

    
    train_f1_perClass = f1_score(y_train, y_train_pred, average=None)
    test_f1_perClass = f1_score(y_test, y_test_pred,average=None)
    
    train_f1_average = f1_score(y_train, y_train_pred, average='macro')
    test_f1_average = f1_score(y_test, y_test_pred,average='macro')
    
    metric_table_train.at[algorithm_name, 'Accuracy'] =  train_accuracy
    metric_table_train.at[algorithm_name, 'Precision(average)'] =  train_precision_average
    metric_table_train.at[algorithm_name, 'F1-score(average)'] = train_f1_average
    

    metric_table_test.at[algorithm_name, 'Accuracy'] = test_accuracy
    metric_table_test.at[algorithm_name, 'Precision(average)'] = test_precision_average
    metric_table_test.at[algorithm_name, 'F1-score(average)'] = test_f1_average

import matplotlib.pyplot as plt

# Plotting Accuracy, Precision, and F1-score for training and test sets
metrics = ['Accuracy', 'Precision(average)', 'F1-score(average)']

fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)

for idx, metric in enumerate(metrics):
    # Train data plot
    axes[0].bar(metric_table_train.index, metric_table_train[metric], label=metric)
    axes[0].set_title(f'Train Data - {metric}')
    axes[0].set_xlabel('Algorithms')
    axes[0].set_ylabel(metric)
    axes[0].tick_params(axis='x', rotation=45)

    # Test data plot
    axes[1].bar(metric_table_test.index, metric_table_test[metric], label=metric)
    axes[1].set_title(f'Test Data - {metric}')
    axes[1].set_xlabel('Algorithms')
    axes[1].tick_params(axis='x', rotation=45)

fig.tight_layout()
plt.show()


print("Metrics - Train Data:\n")
print(metric_table_train.to_string())
print("-------------------------------------------------")

print("Metrics - Test Data:\n")
print(metric_table_test.to_string())


import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier


test_path='testdata'
for filename in os.listdir(test_path):
    if filename.endswith(".csv"):
        # Load each file
        file_path = os.path.join(test_path, filename)
        data = pd.read_csv(file_path, header=None)

# Create an empty dictionary to store the statistics for each row as separate columns
        stats_dict = {}

        # Calculate mean, median, and variance for each row and add them to the dictionary
        for i in range(len(data)):
            row_data = data.iloc[i]
            stats_dict[f'mean{i+1}'] = row_data.mean()
            stats_dict[f'median{i+1}'] = row_data.median()
            stats_dict[f'variance{i+1}'] = row_data.var()
            stats_dict[f'std_dev{i}'] = row_data.std()
            stats_dict[f'skewness{i}'] = row_data.skew()
            stats_dict[f'kurtosis{i}'] = row_data.kurtosis()
            
                        
                
        stats_dict = {k: [v] for k, v in stats_dict.items()} 
        # Convert the dictionary to a DataFrame
        stats_df = pd.DataFrame(stats_dict)
        # print(stats_df)
        stats_df = scaler.transform(stats_df)
        # Train the model
        model = LogisticRegression()
        model.fit(X_train, y_train)  # Ensure X_train and y_train are from your main dataset

        # Predict using the first two rows of stats_df
        input_data = stats_df[:2]  # Selecting the first two rows
        y_pred = model.predict(input_data)
        print(f'{filename}:{y_pred[0]}')
        
       

#asha 0
#Bhav 5
#KK 1
# Lavni 4
#MJ2
#NA3