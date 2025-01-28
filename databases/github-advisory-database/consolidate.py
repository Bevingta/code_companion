import os
import csv
import json

def extract_data_from_repo(repo_path):
    data = []
    for root, dirs, files in os.walk(repo_path):
        for file in files:
            if file.endswith(".json"):
                file_path = os.path.join(root, file)
                with open(file_path, 'r') as f:
                    content = json.load(f)
                    data.append(content)
    return data

def save_data_to_csv(data, csv_file):
    if not data:
        print("No data to save.")
        return
    
    keys = data[0].keys()
    with open(csv_file, 'w', newline='') as f:
        dict_writer = csv.DictWriter(f, fieldnames=keys)
        dict_writer.writeheader()
        dict_writer.writerows(data)

repo_path = "advisory-database"
csv_file = "output.csv"

data = extract_data_from_repo(repo_path)
save_data_to_csv(data, csv_file)

print(f"Data from the repository has been saved to {csv_file}.")
