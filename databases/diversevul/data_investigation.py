import json

with open('diversevul_file.json') as f:
    data = json.load(f)

print(data.keys())
