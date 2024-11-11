import os
import csv
import assemblyai as aai

aai.settings.api_key = ""

# Function to transcribe a file and return the transcript
def transcribe_file(file_url, config):
    transcriber = aai.Transcriber(config=config)
    transcript = transcriber.transcribe(file_url)
    return transcript.text

# Initialize a list to store the transcripts
transcripts = []

# List of folder paths
folder_paths = ['C:\\Users\\carlo\\OneDrive\\Documentos\\NLP-TCC-2024\\04-API-ASSEMBLY\\Falas-RG']

# Configuration for transcription
config = aai.TranscriptionConfig(language_code="pt")

# Iterate through the folders
for folder_path in folder_paths:
    for filename in os.listdir(folder_path):
        if filename.endswith(".mp3"):  # Adjust the file extension if needed
            file_url = os.path.join(folder_path, filename)
            transcript = transcribe_file(file_url, config)
            transcripts.append(transcript)
            print(transcript)

# Specify the output CSV file
output_csv_file = 'transcripts-rg.csv'

# Write the transcripts to the CSV file with UTF-8 encoding
with open(output_csv_file, 'w', newline='', encoding='utf-8') as csvfile:
    csv_writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    for transcript in transcripts:
        csv_writer.writerow([transcript])

print(f"Transcripts saved to {output_csv_file}")
