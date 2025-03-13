import json
import os

def format_for_mistral(input_file, output_file):
    """
    Format JSON data from input file to Mistral fine-tuning format.
    
    Args:
        input_file (str): Path to the input .jsonl file
        output_file (str): Path to save the formatted output file
    """
    formatted_data = []
    
    with open(input_file, 'r') as f:
        for line in f:
            # Parse the JSON line
            data = json.loads(line.strip())
            
            # Extract the components
            instruction = data.get('instruction', '')
            code_input = data.get('input', '')
            output = data.get('output', '')
            
            # Format for Mistral fine-tuning
            # Using the format: <s>[INST] {instruction} {input} [/INST] {output} </s>
            formatted_entry = {
                "text": f"<s>[INST] {instruction}\n\n{code_input} [/INST] {output} </s>"
            }
            
            formatted_data.append(formatted_entry)
    
    # Write the formatted data to the output file
    with open(output_file, 'w') as f:
        for entry in formatted_data:
            f.write(json.dumps(entry) + '\n')
    
    print(f"Processed {len(formatted_data)} entries and saved to {output_file}")
    return formatted_data

# Example usage
if __name__ == "__main__":
    input_file = "llama_formatted.jsonl"  # Your input JSONL file
    output_file = "mistral_formatted_data.jsonl"   # Output file for Mistral
    
    # Process the file
    formatted_data = format_for_mistral(input_file, output_file)
    
    # Print a sample of the first formatted entry
    if formatted_data:
        print("\nSample formatted entry:")
        print(json.dumps(formatted_data[0], indent=2))
