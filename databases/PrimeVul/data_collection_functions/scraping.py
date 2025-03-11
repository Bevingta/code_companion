from scrape_cvss import get_cvss_info
import json

#* LOAD DATASET *#
dataset = "train"
path = f"primevul_{dataset}.jsonl"

data = [] # Contains dataset

# Open the JSONL file
with open(path, 'r') as file:
    for line in file:
        # Parse each line as a JSON object and append to the list
        data.append(json.loads(line))

new_data = []

seen_scores = {}

beginning = 132001
end = len(data)
for i in range(beginning, end):
    print(f"Scraping entry {i} in the database...", end="")
    idx = data[i]["idx"]
    cve_id = data[i]["cve"]
    func = data[i]["func"]

    try:
        cvss_score = seen_scores[cve_id]['cvss_score']
        cvss_severity = seen_scores[cve_id]['cvss_severity']
    except KeyError:
        seen_scores[cve_id] = {} 
        cvss_info = get_cvss_info(cve_id) or {}  # Ensure cvss_info is a dict, not None
        cvss_score = cvss_info.get("score", 0.0)  # Default score to 0.0 if not present
        cvss_severity = cvss_info.get("level", None)
        seen_scores[cve_id]['cvss_score'] = cvss_score
        seen_scores[cve_id]['cvss_severity'] = cvss_severity


    print({"idx": idx, "cve_id": cve_id, "cvss_score": cvss_score, "cvss_severity": cvss_severity})
    new_data.append({"idx": idx, "func": func, "cve_id": cve_id, "cvss_score": cvss_score, "cvss_severity": cvss_severity})

#* SAVE DATA
output_file = f"entries_{beginning}_to_{end}_{dataset}.json"
try:
    with open(output_file, 'r', encoding='utf-8') as file:
        existing_data = json.load(file)  # Load the existing array
except FileNotFoundError:
    existing_data = []  # Initialize as empty if the file does not exist

# Append new data to the existing array
existing_data.extend(new_data)

# Write the updated array back to the file
with open(output_file, 'w', encoding='utf-8') as file:
    json.dump(existing_data, file, indent=4)

print(f"Data successfully saved entries from position {beginning} to {end} to file: '{output_file}'")
