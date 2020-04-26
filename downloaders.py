# Credit: https://stackoverflow.com/a/39225039
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import requests
from pathlib import Path
import time


def download_file_from_google_drive(id, destination):
    """Download files from google drive as bytes and write it to some file
    
    Arguments:
        id {string} -- file id on google drive
        destination {string} -- file on disk to write stream of download data to
    """

    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith("download_warning"):
                return value

        return None

    def save_response_content(response, destination):
        CHUNK_SIZE = 32768
        with open(destination, "wb") as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)
    try:
        URL = "https://docs.google.com/uc?export=download"

        session = requests.Session()

        response = session.get(URL, params={"id": id}, stream=True)
        token = get_confirm_token(response)

        if token:
            params = {"id": id, "confirm": token}
            response = session.get(URL, params=params, stream=True)

        save_response_content(response, destination)
    except:
        print("Download error. Retry in 5 sec.")
        time.sleep(5)
        download_file_from_google_drive(id,destination)
