import json
from scrape_cvss import get_cvss_info

# File paths from argument
cleaned_db = "all_validation_dataset.json" #input("Please enter the name of the file you would like to verify (omit extension): ") + ".json"
dataset = "valid"#input("Please enter the name of the primevul dataset used in the file above (simply enter 'train', 'test', or 'valid'): ")

# Load the existing array from the specified file
data = []
with open(cleaned_db, 'r', encoding='utf-8') as file:
    data = json.load(file)

n = len(data)
all_negatives = []
for i in range(n):
    cvss = data[i]["cvss_score"]
    idx = data[i]["idx"]
    if cvss == 0.0:
        all_negatives.append(idx)

# Load the primevul database
path = f"primevul_{dataset}.jsonl"
database = []

with open(path, 'r') as file:
    for line in file:
        database.append(json.loads(line))

database_len = len(database)
tracker = 0
false_negatives = []
for idx in all_negatives:
    for i in range(tracker, database_len):
        current_db_idx = database[i]["idx"]
        if current_db_idx == idx:
            tracker = i
            cve_id = database[i]["cve"]
            if cve_id != "None":
                false_negatives.append(idx)
            break

fn_len = len(false_negatives)
print(f"There are {fn_len} false negatives.")
if fn_len != 0:
    do_repair = input("Would you like to repair false negatives? Please type 'y' if so: ")
    if do_repair == "y":
        output_file = input("Please enter the name of the output file you would like (omit extension): ") + ".json"
        print("Fixing false negatives...")
        # Fix false negatives
        fn_pos = 0
        fn_idx = false_negatives[fn_pos]
        fixed_data = []
        for entry in range(n):
            current_entry_idx = data[entry]["idx"]
            if current_entry_idx == fn_idx and fn_pos < fn_len:
                cvss_info = get_cvss_info(cve_id)
                cvss_score = cvss_info.get("score", 0.0)  # Provide a default score of 0.0
                cvss_severity = cvss_info.get("level", "Unknown")
                func = data[entry]["func"]
                fixed_data.append({
                    "idx": fn_idx,
                    "func": func,
                    "cve_id": cve_id,
                    "cvss_score": cvss_score,
                    "cvss_severity": cvss_severity
                })
                print(f"Fixed element with idx {fn_idx}")
                fn_pos += 1
                if fn_pos < fn_len:
                    fn_idx = false_negatives[fn_pos]
            else:
                fixed_data.append(data[entry])

        with open(output_file, 'w', encoding='utf-8') as file:
            json.dump(fixed_data, file, indent=4)

        print(f"Successfully fixed {fn_len} false negative functions and uploaded recompiled file as {output_file}")