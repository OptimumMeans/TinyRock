import subprocess
import json
import os
import sys

EXIFTOOL_PATH = r"C:\exiftool.exe"  # Update this if your ExifTool is in a different location

def check_exiftool():
    try:
        subprocess.run([EXIFTOOL_PATH, '-ver'], capture_output=True, check=True)
        return True
    except subprocess.CalledProcessError:
        print("ExifTool is installed but returned an error.")
        return False
    except FileNotFoundError:
        print(f"ExifTool not found at {EXIFTOOL_PATH}. Please ensure the path is correct.")
        return False

def extract_metadata(file_path):
    print(f"Extracting metadata for: {file_path}")
    try:
        result = subprocess.run([EXIFTOOL_PATH, '-j', file_path], capture_output=True, text=True, check=True)
        try:
            metadata = json.loads(result.stdout)
            return metadata[0] if metadata else {}
        except json.JSONDecodeError:
            print(f"Error decoding JSON for {file_path}: {result.stdout}")
            return {}
    except subprocess.CalledProcessError as e:
        print(f"Error running ExifTool: {e.stderr}")
        return {}

def process_directory(directory_path, output_file):
    if not check_exiftool():
        return

    all_metadata = {}
    file_count = 0
    
    if not os.path.exists(directory_path):
        print(f"Error: The path {directory_path} does not exist.")
        return

    if os.path.isfile(directory_path):
        if directory_path.lower().endswith(('.jpg', '.jpeg', '.mp4', '.mov', '.dng')):
            metadata = extract_metadata(directory_path)
            if metadata:
                all_metadata[os.path.basename(directory_path)] = metadata
                file_count += 1
    else:
        for root, _, files in os.walk(directory_path):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.mp4', '.mov', '.dng')):
                    file_path = os.path.join(root, file)
                    metadata = extract_metadata(file_path)
                    if metadata:
                        all_metadata[file] = metadata
                        file_count += 1
    
    print(f"Processed {file_count} files")
    
    if not all_metadata:
        print("No metadata was extracted from any files.")
    else:
        with open(output_file, 'w') as f:
            json.dump(all_metadata, f, indent=2)
        print(f"Metadata extraction complete. Check {output_file}")

# Example usage
directory_path = os.path.join(os.getcwd(), 'img', 'test1.jpg')  # Update this path to your DJI file or directory
output_file = 'dji_metadata.json'
process_directory(directory_path, output_file)