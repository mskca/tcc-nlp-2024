import os
import json

# Path to the directory with .txt files
directory_path = 'C:\\Users\\carlo\\OneDrive\\Documentos\\NLP-TCC-2024\\03-API-ELEVENLABS-VOICE\\Roteiros'

# Expected structure keys
required_keys = {
    "falas": [
        {
            "nome": str,
            "genero": str,
            "tipo": str,
            "fala": str
        }
    ]
}

# Function to validate the JSON structure
def validate_json_structure(data, required_structure):
    if isinstance(data, dict) and "falas" in data:
        falas = data["falas"]
        if isinstance(falas, list):
            for fala in falas:
                for key, key_type in required_structure["falas"][0].items():
                    if key not in fala or not isinstance(fala[key], key_type):
                        return False
            return True
    return False

# Function to open file with multiple encoding attempts
def open_file_with_encoding(file_path):
    try:
        # Try to open with UTF-8
        with open(file_path, 'r', encoding='utf-8') as file:
            return json.load(file)
    except (UnicodeDecodeError, json.JSONDecodeError):
        try:
            # Try to open with Latin-1 if UTF-8 fails
            with open(file_path, 'r', encoding='latin-1') as file:
                return json.load(file)
        except (UnicodeDecodeError, json.JSONDecodeError):
            return None

# Loop through all the files in the directory
def check_files(directory):
    incorrect_files = []
    
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            file_path = os.path.join(directory, filename)
            data = open_file_with_encoding(file_path)
            if data is None:
                incorrect_files.append(filename)
            elif not validate_json_structure(data, required_keys):
                incorrect_files.append(filename)
    
    # Summary output
    if incorrect_files:
        print("The following files are either not valid JSON or have incorrect structure:")
        for file in incorrect_files:
            print(f"- {file}")
    else:
        print("All files are correctly structured and valid.")

# Run the check
check_files(directory_path)
