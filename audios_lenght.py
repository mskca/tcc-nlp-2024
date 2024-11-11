import os
from mutagen.mp3 import MP3

def get_total_audio_length_and_file_count(directory):
    total_length = 0
    file_count = 0
    # Loop through all files in the directory
    for filename in os.listdir(directory):
        # Check if the file is an MP3
        if filename.endswith(".mp3"):
            file_count += 1
            file_path = os.path.join(directory, filename)
            # Load the MP3 file using Mutagen
            audio = MP3(file_path)
            # Add the audio length (in seconds) to the total length
            total_length += audio.info.length

    # Convert the total length from seconds to hours, minutes, and seconds
    hours = int(total_length // 3600)
    minutes = int((total_length % 3600) // 60)
    seconds = int(total_length % 60)

    return hours, minutes, seconds, file_count

# Directory containing the MP3 files
directory = "C:\\Users\\carlo\\OneDrive\\Documentos\\NLP-TCC-2024\\03-API-ELEVENLABS-VOICE\\Falas"

hours, minutes, seconds, file_count = get_total_audio_length_and_file_count(directory)

print(f"Total number of MP3 files: {file_count}")
print(f"Total length of MP3 files: {hours} hours, {minutes} minutes, {seconds} seconds")
