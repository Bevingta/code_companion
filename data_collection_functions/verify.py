import json
from scrape_cvss import get_cvss_info

# File paths from argument
cleaned_db = "all_validation_dataset.json" #input("Please enter the name of the file you would like to verify (omit extension): ") + ".json"
dataset = "valid"#input("Please enter the name of the primevul dataset used in the file above (simply enter 'train', 'test', or 'valid'): ")

# Load the existing array from the specified file
reformatted_data = []
with open(cleaned_db, 'r', encoding='utf-8') as file:
    reformatted_data = json.load(file)

reformatted_data_len = len(reformatted_data)
all_negatives = []
for i in range(reformatted_data_len):
    cvss = reformatted_data[i]["cvss_score"]
    idx = reformatted_data[i]["idx"]
    if cvss == 0.0:
        all_negatives.append(idx)

# Load the primevul database
path = f"primevul_{dataset}.jsonl"
primevul = []

with open(path, 'r') as file:
    for line in file:
        primevul.append(json.loads(line))

primevul_len = len(primevul)
tracker = 0
false_negatives = []
for idx in all_negatives:
    for i in range(tracker, primevul_len):
        current_db_idx = primevul[i]["idx"]
        if current_db_idx == idx:
            tracker = i
            cve_id = primevul[i]["cve"]
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
        fn_pos = 0 # position in the false_negatives array
        fn_idx = false_negatives[fn_pos] # idx of false_negative
        fixed_data = []
        for i in range(0, reformatted_data_len): # loop through every entry in reformatted database
            current_entry_idx = reformatted_data[i]["idx"]
            if current_entry_idx == fn_idx and fn_pos < fn_len:
                cve_id = reformatted_data[i]["cve_id"]
                cvss_info = get_cvss_info(cve_id) or {}
                cvss_score = cvss_info.get("score", 0.0)  # Provide a default score of 0.0
                cvss_severity = cvss_info.get("level", None)
                func = reformatted_data[i]["func"]
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
                fixed_data.append(reformatted_data[i])

        with open(output_file, 'w', encoding='utf-8') as file:
            json.dump(fixed_data, file, indent=4)

        print(f"Successfully fixed {fn_len} false negative functions and uploaded recompiled file as {output_file}")