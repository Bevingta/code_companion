import json
import requests
from typing import List, Dict, Any

def chunk_list(lst: List[Any], chunk_size: int) -> List[List[Any]]:
    """Split a list into smaller chunks of specified size."""
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]

def upload_chunk_to_firebase(chunk: List[Dict], index: int, firebase_url: str) -> None:
    """Upload a single chunk to Firebase."""
    chunk_url = f"{firebase_url}/test/chunk_{index}.json"
    response = requests.put(
        chunk_url,
        json=chunk,
        headers={'Content-Type': 'application/json'}
    )
    response.raise_for_status()
    print(f"Uploaded chunk {index} successfully")

def upload_to_firebase(file_path: str, firebase_url: str, chunk_size: int = 1000):
    """Upload JSONL file to Firebase in chunks."""
    try:
        # Read and parse the JSONL file
        json_data = []
        with open(file_path, 'r') as file:
            for line in file:
                if line.strip():  # Skip empty lines
                    json_data.append(json.loads(line))
        
        # Split data into chunks
        chunks = chunk_list(json_data, chunk_size)
        total_chunks = len(chunks)
        print(f"Splitting data into {total_chunks} chunks")
        
        # Create metadata entry
        metadata = {
            "total_chunks": total_chunks,
            "total_records": len(json_data),
            "chunk_size": chunk_size
        }
        
        # Upload metadata
        metadata_url = f"{firebase_url}/test/metadata.json"
        requests.put(
            metadata_url,
            json=metadata,
            headers={'Content-Type': 'application/json'}
        ).raise_for_status()
        print("Uploaded metadata successfully")
        
        # Upload each chunk
        for i, chunk in enumerate(chunks):
            try:
                upload_chunk_to_firebase(chunk, i, firebase_url)
            except requests.exceptions.RequestException as e:
                print(f"Error uploading chunk {i}: {e}")
                print("Continuing with next chunk...")
                continue
        
        print(f"Upload complete. {total_chunks} chunks uploaded.")
        return True
        
    except FileNotFoundError:
        print(f"Error: File {file_path} not found")
        raise
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in file at line {e.lineno}: {e.msg}")
        raise
    except requests.exceptions.RequestException as e:
        print(f"Error during upload: {e}")
        raise

if __name__ == "__main__":
    # Configuration
    FIREBASE_URL = "https://code-companion-8f884-default-rtdb.firebaseio.com"
    FILE_PATH = "databases/primevul/primevul_valid.jsonl"
    CHUNK_SIZE = 1000  # Adjust this based on your data size
    
    try:
        upload_to_firebase(FILE_PATH, FIREBASE_URL, CHUNK_SIZE)
    except Exception as e:
        print(f"Upload failed: {e}")
