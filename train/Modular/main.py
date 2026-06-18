import pandas as pd
import os
from Load_data import load_data
from Train_evaluate_LR import train_and_eval

data_dir = os.path.abspath(os.path.join(os.path.expanduser("~"), "Desktop", "TEAM - Dataset", "data"))
names_file_path = os.path.join(data_dir, "dataset_names.txt")

with open(names_file_path, "r") as f:
    dataset_names = [line.strip() for line in f.readlines() if line.strip()]

# This list will hold a dictionary for each dataset
results = []

for i, name in enumerate(dataset_names):
    file_path = os.path.join(data_dir, str(i + 1), "data.mat")
    
    if os.path.exists(file_path):
        try:
            X, y = load_data(file_path)
            model, metrics, imp = train_and_eval(X, y)
            
            # Create a dictionary for this row
            row = {'Dataset': name}
            row.update(metrics) # Adds Accuracy, F1-Score, AUC-ROC to the row
            results.append(row)
            
            print(f"Processed: {name}")
        except Exception as e:
            print(f"Error on {name}: {e}")

# Convert list of dictionaries to a DataFrame
performance_df = pd.DataFrame(results)

# Save to one master CSV
performance_df.to_csv("model_performance_summary.csv", index=False)

print("\n--- Final Performance Table ---")
print(performance_df)
print("\nSuccess! Master file 'model_performance_summary.csv' created.")