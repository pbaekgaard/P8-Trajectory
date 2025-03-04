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
ROADNET_URL = "https://download.geofabrik.de/asia/china/beijing-latest-free.shp.zip"

# Cache directory
CACHE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/.cache"))
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/raw"))
ROAD_DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/roadnet"))
# Derive a filename from the URL using a hash
CACHE_FILENAME_TDRIVE = os.path.join(CACHE_DIR, hashlib.md5(ZIP_URL.encode()).hexdigest() + ".zip")
CACHE_FILENAME_ROADNET = os.path.join(CACHE_DIR, hashlib.md5(ROADNET_URL.encode()).hexdigest() + ".zip")

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

def download_with_spinner(downloadTdrive = True, downloadRoadnet = True):
    """Download a file with a spinner animation and save to cache."""
    # Create cache directory if it doesn't exist
    os.makedirs(CACHE_DIR, exist_ok=True)
    zip_data = None
    road_net_zip = None
    
    if (downloadTdrive):
        stop_spinner = threading.Event()
        spinner_thread = threading.Thread(target=spinner_animation, args=(stop_spinner,))
        print(f"Downloading T-Drive Dataset")
        spinner_thread.start()
        try:
            with urllib.request.urlopen(ZIP_URL) as response:
                zip_data = response.read()
                
                # Save the downloaded data to the cache file
                with open(CACHE_FILENAME_TDRIVE, 'wb') as cache_file:
                    cache_file.write(zip_data)
                    
            stop_spinner.set()
            spinner_thread.join()
        except Exception as e:
            stop_spinner.set()
            spinner_thread.join()
            raise e

    if (downloadRoadnet):
        stop_spinner = threading.Event()
        spinner_thread = threading.Thread(target=spinner_animation, args=(stop_spinner,))
        print(f"Downloading Roadnet Data")
        spinner_thread.start()
        try:
            with urllib.request.urlopen(ROADNET_URL) as response:
                road_net_zip = response.read()
                
                # Save the downloaded data to the cache file
                with open(CACHE_FILENAME_ROADNET, 'wb') as cache_file:
                    cache_file.write(road_net_zip)
                    
            stop_spinner.set()
            spinner_thread.join()
        except Exception as e:
            stop_spinner.set()
            spinner_thread.join()
            raise e
    return (zip_data, road_net_zip)

def get_zip_data():
    """Get zip data, either from cache or by downloading."""
    tdrive, roadnet = None, None

    # Try loading T-Drive from cache
    if os.path.exists(CACHE_FILENAME_TDRIVE):
        print(f"Using cached zip file for T-Drive from {CACHE_FILENAME_TDRIVE}")
        with open(CACHE_FILENAME_TDRIVE, 'rb') as cache_file_tdrive:
            tdrive = cache_file_tdrive.read()
    
    # If not in cache, download it
    if tdrive is None:
        print("T-Drive not found in cache, downloading...")
        tdrive, _ = download_with_spinner(True, False)
        if tdrive is None:  # Final safeguard
            raise RuntimeError("Failed to retrieve T-Drive dataset!")

    # Try loading Road Network from cache
    if os.path.exists(CACHE_FILENAME_ROADNET):
        print(f"Using cached zip file for Road Network from {CACHE_FILENAME_ROADNET}")
        with open(CACHE_FILENAME_ROADNET, 'rb') as cache_file_roadnet:
            roadnet = cache_file_roadnet.read()
    
    # If not in cache, download it
    if roadnet is None:
        print("Road Network not found in cache, downloading...")
        _, roadnet = download_with_spinner(False, True)
        if roadnet is None:  # Final safeguard
            raise RuntimeError("Failed to retrieve Road Network dataset!")

    return tdrive, roadnet


def extract_from_release_folder(zf, target_dir, roadnet: bool):
    """Extract files from the zip archive to the target directory.
    If roadnet is True, extract only files containing 'roads' in their name.
    If roadnet is False, only extract files from the 'release/' folder.
    Skip extracting files that already exist to speed up the process."""
    os.makedirs(target_dir, exist_ok=True)
    
    file_list = zf.namelist()
    
    if roadnet:
        extract_files = [f for f in file_list if 'roads' in f]
    else:
        extract_files = [f for f in file_list if f.startswith('release/')]
        
    if not extract_files:
        print("Warning: No files found to extract.")
        return
    
    progress = ProgressBar(len(extract_files))
    
    for file_path in extract_files:
        if file_path.endswith('/'):
            continue  # Skip directories
        
        if not roadnet:
            file_path_clean = file_path.replace('release/', '', 1)
        else:
            file_path_clean = file_path
        
        new_path = os.path.join(target_dir, file_path_clean)
        
        # Skip extraction if file already exists
        if os.path.exists(new_path):
            progress.update(file_path)
            continue
        
        os.makedirs(os.path.dirname(new_path), exist_ok=True)
        
        with zf.open(file_path) as source, open(new_path, 'wb') as target:
            target.write(source.read())
        
        progress.update(file_path)
    
    progress.finish()

def main():
    try:
        # Get zip data (either from cache or by downloading)
        (tdrive, roadnet) = get_zip_data()
        
        # Create a BytesIO object from the data
        tdrive = io.BytesIO(tdrive)
        roadnet = io.BytesIO(roadnet)
        
        # Open the zip file and extract from release folder
        with zipfile.ZipFile(tdrive) as td:
            extract_from_release_folder(td, DATA_DIR, False)

        with zipfile.ZipFile(roadnet) as rn:
            extract_from_release_folder(rn, ROAD_DATA_DIR, True)
        
        print("Files extracted successfully into 'data' folder.")
    except zipfile.BadZipFile as e:
        print(f"Error: The zip file is corrupted: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
