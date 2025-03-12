#!/bin/python3
import hashlib
import io
import os
import sys
import threading
import time
import urllib.request
import zipfile

# Placeholder URL for the zip file (replace with the actual URL)
ZIP_URL = "https://www.kaggle.com/api/v1/datasets/download/arashnic/tdriver"

# Cache directory
CACHE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.cache"))
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data"))
# Derive a filename from the URL using a hash
CACHE_FILENAME = os.path.join(CACHE_DIR, hashlib.md5(ZIP_URL.encode()).hexdigest() + ".zip")

def spinner_animation(stop_event):
    """Display a spinner animation in the console."""
    spinner = ['-', '\\', '|', '/']
    i = 0
    while not stop_event.is_set():
        sys.stdout.write(f"\rDownloading... {spinner[i % len(spinner)]}")
        sys.stdout.flush()
        time.sleep(0.1)
        i += 1
    sys.stdout.write("\rDownload complete!       \n")

class ProgressBar:
    """A simple progress bar for tracking extraction progress."""
    def __init__(self, total_files):
        self.total = total_files
        self.current = 0
        self.bar_length = 50

    def update(self, filename):
        self.current += 1
        filled_length = int(self.bar_length * self.current / self.total)
        bar = '█' * filled_length + '-' * (self.bar_length - filled_length)
        percentage = int(100 * self.current / self.total)
        sys.stdout.write(f'\rExtracting: [{bar}] {percentage}% {self.current}/{self.total} - {os.path.basename(filename)}')
        sys.stdout.flush()

    def finish(self):
        sys.stdout.write('\nExtraction complete!\n')

def download_with_spinner():
    """Download a file with a spinner animation and save to cache."""
    # Create cache directory if it doesn't exist
    os.makedirs(CACHE_DIR, exist_ok=True)
    
    stop_spinner = threading.Event()
    spinner_thread = threading.Thread(target=spinner_animation, args=(stop_spinner,))
    spinner_thread.start()
    
    try:
        with urllib.request.urlopen(ZIP_URL) as response:
            zip_data = response.read()
            
            # Save the downloaded data to the cache file
            with open(CACHE_FILENAME, 'wb') as cache_file:
                cache_file.write(zip_data)
                
        stop_spinner.set()
        spinner_thread.join()
        return zip_data
    except Exception as e:
        stop_spinner.set()
        spinner_thread.join()
        raise e

def get_zip_data():
    """Get zip data, either from cache or by downloading."""
    # Check if file exists in cache
    if os.path.exists(CACHE_FILENAME):
        print(f"Using cached zip file from {CACHE_FILENAME}")
        with open(CACHE_FILENAME, 'rb') as cache_file:
            return cache_file.read()
    else:
        return download_with_spinner()

def extract_from_release_folder(zf, target_dir):
    """Extract files from the 'release' folder to the target directory."""
    # Create target directory if it doesn't exist
    os.makedirs(target_dir, exist_ok=True)
    
    # Get all files in the zip
    file_list = zf.namelist()
    
    # Filter files to only include those in the release folder
    release_files = [f for f in file_list if f.startswith('release/')]
    
    if not release_files:
        print("Warning: No files found in 'release' folder.")
        return
    
    # Setup progress bar
    progress = ProgressBar(len(release_files))
    
    # Extract each file, removing the 'release/' prefix
    for file_path in release_files:
        # Skip the release directory itself
        if file_path == 'release/' or file_path.endswith('/'):
            continue
            
        # Remove 'release/' prefix to get the new path
        new_path = os.path.join(target_dir, file_path.replace('release/', '', 1))
        
        # Create the directory structure if needed
        os.makedirs(os.path.dirname(new_path), exist_ok=True)
        
        # Extract the file to the new path
        with zf.open(file_path) as source, open(new_path, 'wb') as target:
            target.write(source.read())
            
        progress.update(file_path)
    
    progress.finish()

def main():
    if os.path.exists(DATA_DIR): return

    try:
        # Get zip data (either from cache or by downloading)
        zip_data = get_zip_data()
        
        # Create a BytesIO object from the data
        zip_file = io.BytesIO(zip_data)
        
        # Open the zip file and extract from release folder
        with zipfile.ZipFile(zip_file) as zf:
            extract_from_release_folder(zf, DATA_DIR)
        
        print("Files extracted successfully into 'data' folder.")
    except urllib.error.URLError as e:
        print(f"Error downloading the zip file: {e}")
    except zipfile.BadZipFile as e:
        print(f"Error: The zip file is corrupted: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
