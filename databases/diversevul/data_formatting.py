import json
import re

def fix_and_parse_json_line(line):
    """Attempt to fix common JSON string issues and parse the line"""
    try:
        # First try parsing as-is
        return json.loads(line)
    except json.JSONDecodeError:
        try:
            # Fix potential unterminated strings in the code
            # This uses a regex to ensure all quotes are properly escaped
            fixed_line = line
            
            # Find the "func" field and properly escape its content
            func_match = re.match(r'({[^}]*"func":\s*")(.*?)(".*})', fixed_line)
            if func_match:
                prefix, code, suffix = func_match.groups()
                # Escape quotes in code
                escaped_code = code.replace('"', '\\"').replace('\n', '\\n')
                fixed_line = prefix + escaped_code + suffix
            
            return json.loads(fixed_line)
        except (json.JSONDecodeError, Exception) as e:
            raise Exception(f"Failed to parse line: {str(e)}")

def parse_file_safely(filename):
    valid_entries = []
    errors = []
    
    with open(filename, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:  # Skip empty lines
                continue
            
            try:
                entry = fix_and_parse_json_line(line)
                valid_entries.append(entry)
            except Exception as e:
                errors.append(f"Line {line_num}: {str(e)}")
                # Optionally save problematic lines for inspection
                with open('problematic_lines.txt', 'a') as err_f:
                    err_f.write(f"Line {line_num}:\n{line}\n\n")

    return valid_entries, errors

# Let's also add a function to inspect problematic lines
def inspect_line(filename, line_number, context=2):
    """Show the content around a problematic line"""
    lines = []
    start = max(1, line_number - context)
    end = line_number + context
    
    with open(filename, 'r') as f:
        for i, line in enumerate(f, 1):
            if start <= i <= end:
                lines.append(f"Line {i}: {line[:200]}...")  # Show first 200 chars
            if i > end:
                break
    
    return lines

# Try parsing the file
try:
    entries, errors = parse_file_safely('diversevul_raw.json')
    
    print(f"Successfully parsed {len(entries)} entries")
    if errors:
        print("\nFound errors:")
        for error in errors[:10]:  # Show first 10 errors
            print(error)
        
        # Inspect the first error more closely
        if errors:
            error_line_num = int(errors[0].split(':')[0].split()[1])
            print("\nContext around first error:")
            context_lines = inspect_line('diversevul_file.json', error_line_num)
            for line in context_lines:
                print(line)
            
except Exception as e:
    print(f"Error: {str(e)}")
