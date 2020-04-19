from pathlib import Path
from downloaders import download_file_from_google_drive
from zipfile import ZipFile
import json
from PIL import Image
import glob

# Static variables
FILE_ID = "1vAtykMfhKDT1LywJlpSE9DcxjvNJdBVL"
PATH_TO_STORE_DATA = "./data/"
FILE_NAME = "data.zip"


def load_data():
    """Load data to PIL images to dictionary of group->name->PIL Images
    """
    # Download images
    download_data()

    # Unzip them
    with ZipFile(PATH_TO_STORE_DATA + FILE_NAME, "r") as zip_ref:
        zip_ref.extractall(PATH_TO_STORE_DATA)

    # Load all images into program
    with open(PATH_TO_STORE_DATA + "members.json") as f:
        data = json.load(f)

    for group in data.keys():
        data[group] = {name: [] for name in data[group]}

    for group in data.keys():
        for name in data[group].keys():
            images_path = PATH_TO_STORE_DATA + name + "/"
            for image in glob.glob(images_path + "*.*"):
                data[group][name].append(Image.open(image))
    return data


def download_data():
    """Download data zip file from google drive
    """

    if not Path(PATH_TO_STORE_DATA + FILE_NAME).exists():
        print("Downloading", FILE_NAME, "...")

        # Make directory to store downloaded model
        Path(PATH_TO_STORE_DATA).mkdir(
            parents=True, exist_ok=True,
        )

        # Download from google drives
        download_file_from_google_drive(FILE_ID, PATH_TO_STORE_DATA + FILE_NAME)
