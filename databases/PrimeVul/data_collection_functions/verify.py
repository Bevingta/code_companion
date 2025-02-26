# Used ChatGPT to clean up code, however, wrote the code myself.
import json
from scrape_cvss import get_cvss_info

# Get input file and dataset type from user
input_file = input("Please enter the name of the file you would like to verify (omit extension): ") + ".json"
dataset_type = input("Please enter the name of the primevul dataset used in the file above (e.g., 'train', 'test', 'val'): ")

# Load the existing dataset
with open(input_file, 'r', encoding='utf-8') as file:
    dataset = json.load(file)

# Identify potential false negatives (entries with a CVSS score of 0.0)
zero_vulnerability_indices = [entry["idx"] for entry in dataset if entry["cvss_score"] == 0.0]

# Load the corresponding primevul dataset
primevul_path = f"primevul_{dataset_type}.jsonl"
primevul_data = [json.loads(line) for line in open(primevul_path, 'r')]

# Check if zero-vulnerability indices are truly non-vulnerable
false_negatives = []
tracker = 0
for idx in zero_vulnerability_indices:
    for i in range(tracker, len(primevul_data)):
        if primevul_data[i]["idx"] == idx:
            tracker = i
            if primevul_data[i]["cve"] != "None":
                false_negatives.append(idx)
            break

# Output the false negatives found
fn_count = len(false_negatives)
print(f"Found {fn_count} false negatives.")

# Option to repair false negatives
if fn_count > 0 and input("Would you like to repair false negatives? (y/n): ").lower() == 'y':
    output_file = input("Please enter the desired output file name (omit extension): ") + ".json"
    print("Fixing false negatives...")

    fixed_data = []
    fn_index = 0
    for entry in dataset:
        if fn_index < fn_count and entry["idx"] == false_negatives[fn_index]:
            cve_id = primevul_data[tracker]["cve"]
            cvss_info = get_cvss_info(cve_id)
            fixed_data.append({
                "idx": entry["idx"],
                "func": entry["func"],
                "cve_id": cve_id,
                "cvss_score": cvss_info.get("score", 0.0),
                "cvss_severity": cvss_info.get("level", "Unknown")
            })
            print(f"Repaired entry with idx {entry['idx']}")
            fn_index += 1
        else:
            fixed_data.append(entry)

    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(fixed_data, file, indent=4)

    print(f"Successfully fixed {fn_count} false negatives. Output saved as {output_file}")