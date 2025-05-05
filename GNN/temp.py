from utils.json_functions import load_json_array, save_array_to_json
import random

test = load_json_array("data/split_datasets/test.json")
valid = load_json_array("data/split_datasets/valid.json")

safe = [item for item in test if item["target"] == 0]
vulnt = [item for item in test if item["target"] == 1]

vulnv = [item for item in valid if item["target"] == 1]

vuln = vulnt + vulnv

total = len(vuln) + len(safe)
print(len(vuln))

safe = random.sample(safe, 3 * len(vuln))
print(len(safe))

combined = vuln + safe

save_array_to_json(combined, "primevul_for_gpt.json")