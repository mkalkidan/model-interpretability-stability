import pandas as pd
import os
from Load_data import load_data
from Train_evaluate_LR import train_and_eval

# Define paths
data_dir = os.path.abspath(os.path.join(os.getcwd(), "..", "..", "data"))
names_file_path = os.path.join(data_dir, "dataset_names.txt")

print(f"Looking for data in: {data_dir}")

if not os.path.exists(names_file_path):
    print(f"ERROR: Could not find dataset_names.txt at {names_file_path}")
else:
    with open(names_file_path, "r") as f:
        dataset_names = [line.strip() for line in f.readlines() if line.strip()]
    
    print(f"Found {len(dataset_names)} datasets to process.")
    
    performance = pd.DataFrame(columns=['Dataset', 'Accuracy'])

    for i, name in enumerate(dataset_names):
        folder_num = i + 1
        file_path = os.path.join(data_dir, str(folder_num), "data.mat")
        
        if os.path.exists(file_path):
            print(f"Processing {folder_num}: {name}...")
            try:
                X, y = load_data(file_path)
                model, acc = train_and_eval(X, y)
                
                new_row = pd.DataFrame({'Dataset': [name], 'Accuracy': [acc]})
                performance = pd.concat([performance, new_row], ignore_index=True)
                print(f"Success: {name} | Accuracy: {acc:.4f}")
            except Exception as e:
                print(f"Failed to process {name}: {e}")
        else:
            print(f"Skipping: {file_path} not found.")

    print("\n--- Final Performance Summary ---")
    print(performance)
    # performance.to_csv("performance_results.csv", index=False)