import os
import json

# Directory containing the .txt files
directory = 'C:\\Users\\carlo\\OneDrive\\Documentos\\NLP-TCC-2024\\03-API-ELEVENLABS-VOICE\\Roteiros'

# Function to validate the 'genero' key
def validate_genero(json_data):
    # Check if 'falas' is in the JSON and is a list
    if 'falas' in json_data and isinstance(json_data['falas'], list):
        for entry in json_data['falas']:
            # Ensure 'genero' exists and is either 'M' or 'F'
            if 'genero' not in entry or entry['genero'] not in ['M', 'F']:
                return False
        return True
    return False

# Function to try reading the file with different encodings
def read_file_with_encoding(file_path):
    try:
        # Try reading with utf-8 encoding first
        with open(file_path, 'r', encoding='utf-8') as file:
            return json.load(file)
    except UnicodeDecodeError:
        # If utf-8 fails, try latin-1 encoding
        with open(file_path, 'r', encoding='latin-1') as file:
            return json.load(file)

# Track if all files have valid 'genero'
all_valid = True

# Loop through the files in the directory
for filename in os.listdir(directory):
    if filename.endswith('.txt'):
        file_path = os.path.join(directory, filename)
        try:
            # Attempt to read the file with possible encodings
            data = read_file_with_encoding(file_path)
            if not validate_genero(data):
                print(f"Invalid 'genero' in file: {filename}")
                all_valid = False
                break  # Stop checking further if one invalid is found
        except json.JSONDecodeError:
            print(f"Error decoding JSON in file: {filename}")
            all_valid = False
            break
        except Exception as e:
            print(f"Error processing file {filename}: {e}")
            all_valid = False
            break

# If all files have valid 'genero', print the message
if all_valid:
    print("All 'genero' values are valid in all files!")
