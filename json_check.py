import os
import json

# Path to the directory containing your files
directory_path = 'C:\\Users\\carlo\\OneDrive\\Documentos\\NLP-TCC-2024\\03-API-ELEVENLABS-VOICE\\Roteiros'

def validate_json_structure(file_content):
    try:
        # Try to load the content as JSON
        data = json.loads(file_content)
        
        # Check if the JSON contains the key 'falas'
        if "falas" in data and isinstance(data['falas'], list):
            for fala in data["falas"]:
                # Check each field in the fala object
                if not all(key in fala for key in ["nome", "genero", "tipo", "fala"]):
                    return False
                # Optional: Validate the data types of each field
                if not isinstance(fala['nome'], str) or not isinstance(fala['genero'], str) \
                        or not isinstance(fala['tipo'], str) or not isinstance(fala['fala'], str):
                    return False
            return True
        return False
    except json.JSONDecodeError:
        return False

def read_file(file_path):
    # First, try to read the file in utf-8 encoding
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except UnicodeDecodeError:
        # If utf-8 fails, read the file using latin-1 encoding
        with open(file_path, 'r', encoding='latin-1') as file:
            return file.read()

def check_files(directory_path):
    invalid_files = []
    
    # Iterate over each file in the directory
    for filename in os.listdir(directory_path):
        if filename.endswith('.txt'):  # Assuming files have .txt extension
            file_path = os.path.join(directory_path, filename)
            file_content = read_file(file_path)  # Read the file content
            
            if not validate_json_structure(file_content):
                print(f"File '{filename}' does not match the expected structure.")
                invalid_files.append(filename)
    
    # Summary of invalid files
    if invalid_files:
        print(f"\nThe following files do not match the expected structure: {invalid_files}")
    else:
        print("All files are correctly structured.")

# Run the check
check_files(directory_path)
