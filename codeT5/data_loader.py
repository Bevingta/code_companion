import requests
import json
import time
import csv
import re
import sys

# Replace with your actual API key if needed
NVD_API_KEY = "8e783767-9152-4e3b-a4ed-6ccb2505c5b7"

# Create a session for efficiency
session = requests.Session()

def get_cvss_from_nvd(cve_id):
    if not cve_id or cve_id == "None":
        return None  # Skip entry if no CVE ID is present

    base_url = "https://services.nvd.nist.gov/rest/json/cves/2.0?cveId="
    url = f"{base_url}{cve_id}"

    headers = {
        "User-Agent": "Mozilla/5.0"  # Helps avoid being blocked
    }

    max_retries = 5  # Number of retries for failed requests
    delay = 5  # Initial delay (seconds) for backoff

    for attempt in range(max_retries):
        try:
            response = session.get(url, headers=headers, timeout=20)  # ⏳ Increased timeout

            if response.status_code == 200:
                try:
                    data = response.json()
                    vulnerabilities = data.get("vulnerabilities", [])
                    if vulnerabilities:
                        metrics = vulnerabilities[0].get("cve", {}).get("metrics", {})
                        # Extract baseScore from CVSS v3.1, v3.0, or v2
                        for version in ["cvssMetricV31", "cvssMetricV30", "cvssMetricV2"]:
                            if version in metrics and isinstance(metrics[version], list):
                                base_score = metrics[version][0].get("cvssData", {}).get("baseScore")
                                if base_score is not None:
                                    return base_score
                    return None  # No baseScore found
                except json.JSONDecodeError:
                    return None  # JSON parsing error

            elif response.status_code == 404:
                return None  # CVE not found

            elif response.status_code == 503:
                time.sleep(delay)
                delay *= 2  # Exponential backoff

            else:
                return None  # Other API failure

        except requests.exceptions.Timeout:
            time.sleep(delay)
            delay *= 2  # Exponential backoff

        except requests.exceptions.RequestException:
            time.sleep(delay)
            delay *= 2

    return None  # Max retries reached

def extract_cve_id(nvd_url):
    """
    Extracts the CVE ID from a given NVD URL.
    For example, given a URL containing 'CVE-2021-34527', it returns 'CVE-2021-34527'.
    """
    match = re.search(r"(CVE-\d{4}-\d+)", nvd_url)
    if match:
        return match.group(1)
    return None

def process_jsonl_file(input_file, output_csv, limit=None):
    """
    Reads the JSONL file, filters entries, calls the NVD API to get the CVSS score,
    and writes out the 'func' field and score as CSV.
    
    Args:
        input_file (str): Path to the JSONL file.
        output_csv (str): Path for the output CSV file.
        limit (int or None): Maximum number of valid entries to process. If None, all entries are processed.
    """
    count = 0
    with open(output_csv, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["func", "cvss_score"])  # CSV header

        with open(input_file, "r") as infile:
            for line in infile:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue

                # Only process entries where "project" is not "ghostscript" and "nvd_url" is non-empty.
                project = entry.get("project", "").lower()
                nvd_url = entry.get("nvd_url", "").strip()
                if project == "ghostscript" or not nvd_url:
                    continue

                cve_id = extract_cve_id(nvd_url)
                if not cve_id:
                    continue

                cvss_score = get_cvss_from_nvd(cve_id)
                func_field = entry.get("func", "")

                # Write the function field and the corresponding CVSS score (or blank if not found)
                writer.writerow([func_field, cvss_score if cvss_score is not None else ""])
                count += 1

                # If a limit is specified and reached, stop processing further entries.
                if limit is not None and count >= limit:
                    print(f"Processed limit of {limit} entries. Stopping.")
                    break

if __name__ == "__main__":
    # Specify your input JSONL file and desired output CSV file
    input_file = "primevul_train.jsonl"
    output_csv = "output.csv"
    
    # Optionally, pass the limit as a command line argument. For example:
    # python script.py 100
    entry_limit = None
    if len(sys.argv) > 1:
        try:
            entry_limit = int(sys.argv[1])
        except ValueError:
            print("Invalid limit provided. Using no limit.")
            entry_limit = None

    process_jsonl_file(input_file, output_csv, limit=entry_limit)
