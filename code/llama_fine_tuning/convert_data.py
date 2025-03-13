import json
import os
from tqdm import tqdm

def convert_to_llama_format(input_file, output_file):
    """
    Convert JSON data to the format required for LLaMA fine-tuning.
    
    Args:
        input_file (str): Path to the input JSON file
        output_file (str): Path to the output JSONL file for LLaMA fine-tuning
    """
    # Load the JSON data
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_file)
    if output_dir:  # Only try to create directory if there's a directory component
        os.makedirs(output_dir, exist_ok=True)
    
    # Process each item and write to output file
    with open(output_file, 'w') as out_f:
        for item in tqdm(data, desc="Converting data"):
            # Extract relevant fields
            idx = item.get("idx", "")
            func = item.get("func", "")
            cve_id = item.get("cve_id", "")
            cvss_score = item.get("cvss_score", "")
            cvss_severity = item.get("cvss_severity", "")
            
            # Create instruction for the model
            instruction = "Analyze the following code for security vulnerabilities and assign a CVSS score:"
            
            # Create the input for the model
            input_text = func
            
            # Create the expected output
            output = f"Vulnerability detected: {cve_id}\nCVSS Score: {cvss_score}\nSeverity: {cvss_severity}\n"
            
            # LLaMA format requires a specific JSON structure
            llama_format = {
                "instruction": instruction,
                "input": input_text,
                "output": output
            }
            
            # Write to the output file
            out_f.write(json.dumps(llama_format) + '\n')
    
    print(f"Conversion complete. Output saved to {output_file}")

def convert_to_alpaca_format(input_file, output_file):
    """
    Convert JSON data to the Alpaca format, which is commonly used for LLaMA fine-tuning.
    
    Args:
        input_file (str): Path to the input JSON file
        output_file (str): Path to the output JSONL file for Alpaca format
    """
    # Load the JSON data
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_file)
    if output_dir:  # Only try to create directory if there's a directory component
        os.makedirs(output_dir, exist_ok=True)
    
    # Process each item and write to output file
    with open(output_file, 'w') as out_f:
        for item in tqdm(data, desc="Converting to Alpaca format"):
            # Extract relevant fields
            idx = item.get("idx", "")
            func = item.get("func", "")
            cve_id = item.get("cve_id", "")
            cvss_score = item.get("cvss_score", "")
            cvss_severity = item.get("cvss_severity", "")
            
            # Create the Alpaca format entry
            alpaca_format = {
                "instruction": "Analyze the provided code for security vulnerabilities. If found, identify the vulnerability type and assign an appropriate CVSS score and severity level.",
                "input": func,
                "output": f"This code contains a security vulnerability identified as {cve_id}. The vulnerability has a CVSS score of {cvss_score}, which is categorized as {cvss_severity} severity."
            }
            
            # Write to the output file
            out_f.write(json.dumps(alpaca_format) + '\n')
    
    print(f"Conversion to Alpaca format complete. Output saved to {output_file}")

def get_statistics(input_file):
    """
    Get statistics about the dataset.
    
    Args:
        input_file (str): Path to the input JSON file
    """
    # Load the JSON data
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # Count severities
    severity_counts = {}
    cve_counts = {}
    
    for item in data:
        severity = item.get("cvss_severity", "UNKNOWN")
        cve_id = item.get("cve_id", "UNKNOWN")
        
        # Count severities
        if severity in severity_counts:
            severity_counts[severity] += 1
        else:
            severity_counts[severity] = 1
        
        # Count CVEs
        if cve_id in cve_counts:
            cve_counts[cve_id] += 1
        else:
            cve_counts[cve_id] = 1
    
    # Print statistics
    print(f"Total samples: {len(data)}")
    print("\nSeverity distribution:")
    for severity, count in severity_counts.items():
        print(f"  {severity}: {count} ({count/len(data)*100:.2f}%)")
    
    print(f"\nUnique CVEs: {len(cve_counts)}")
    
    # Print the top 5 most common CVEs
    top_cves = sorted(cve_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    print("\nTop 5 most common CVEs:")
    for cve, count in top_cves:
        print(f"  {cve}: {count} samples")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert JSON data to LLaMA fine-tuning format")
    parser.add_argument("--input", required=True, help="Path to the input JSON file")
    parser.add_argument("--output", required=True, help="Path to the output file")
    parser.add_argument("--format", choices=["llama", "alpaca"], default="llama", 
                        help="Output format (llama or alpaca)")
    parser.add_argument("--stats", action="store_true", help="Print dataset statistics")
    
    args = parser.parse_args()
    
    if args.stats:
        get_statistics(args.input)
    
    if args.format == "llama":
        convert_to_llama_format(args.input, args.output)
    else:
        convert_to_alpaca_format(args.input, args.output)
