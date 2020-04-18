from pathlib import Path
from downloaders import download_file_from_google_drive
from zipfile import ZipFile
# Static variables
FILE_ID = "1BQNK9T2j1KeCDMKFydvIl6r5eZMEsVeh"
PATH_TO_STORE_MODEL = "./data/"
FILE_NAME = "data.zip"


def load_data():
    download_data()
    with ZipFile(PATH_TO_STORE_MODEL + FILE_NAME, 'r') as zip_ref:
        zip_ref.extractall(PATH_TO_STORE_MODEL)

def download_data():
    """Download data zip file from google drive
    """

    if not Path(PATH_TO_STORE_MODEL + FILE_NAME).exists():
        print("Downloading",FILE_NAME,"...")

        # Make directory to store downloaded model
        Path(PATH_TO_STORE_MODEL).mkdir(
            parents=True, exist_ok=True,
        )

        # Download from google drives
        download_file_from_google_drive(FILE_ID, PATH_TO_STORE_MODEL + FILE_NAME)

print(load_data())