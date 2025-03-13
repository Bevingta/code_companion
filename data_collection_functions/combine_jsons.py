import json


file_names = ["entries_0_to_44000_train.json", "entries_44001_to_88000_train.json", "entries_88001_to_132000_train.json", "entries_132001_to_175797_train.json"]
output = "all_train_dataset.json"

combined = []

for file in file_names:
    with open(file, 'r', encoding='utf-8') as f:
        contents = json.load(f)  # Load the existing array
    combined.extend(contents)

with open(output, 'w', encoding='utf-8') as file:
    json.dump(combined, file, indent=4)