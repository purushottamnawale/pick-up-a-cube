#!/usr/bin/env python3
"""
Download Franka Emika Panda robot model from mujoco_menagerie.
"""

import os
import urllib.request
import zipfile
import shutil

MENAGERIE_URL = "https://github.com/google-deepmind/mujoco_menagerie/archive/refs/heads/main.zip"
ROBOT_DIR = "franka_emika_panda"

def download_robot_model():
    """Download and extract the Franka Panda model."""
    
    if os.path.exists(ROBOT_DIR):
        print(f"Robot model already exists in '{ROBOT_DIR}/'")
        return
    
    print("Downloading mujoco_menagerie...")
    zip_path = "menagerie.zip"
    urllib.request.urlretrieve(MENAGERIE_URL, zip_path)
    
    print("Extracting Franka Emika Panda model...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall("temp_menagerie")
    
    # Move the Franka robot folder to current directory
    src = os.path.join("temp_menagerie", "mujoco_menagerie-main", "franka_emika_panda")
    shutil.move(src, ROBOT_DIR)
    
    # Cleanup
    shutil.rmtree("temp_menagerie")
    os.remove(zip_path)
    
    print(f"Robot model downloaded to '{ROBOT_DIR}/'")

if __name__ == "__main__":
    download_robot_model()
