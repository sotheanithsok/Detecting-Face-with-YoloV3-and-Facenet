# Speical thank to Hiroki Taniai for providing pretrained facenet model.
# https://github.com/nyoki-mtl/keras-facenet

from pathlib import Path
from downloaders import download_file_from_google_drive
import tensorflow as tf


# Static variables
FILE_ID = "1PZ_6Zsy1Vb0s0JmjEmVd8FS99zoMCiN1"
PATH_TO_STORE_MODEL = "./models/facenet/"
FILE_NAME = "facenet_keras.h5"


def load_model():
    """Load pretrained facenet model
    """

    download_model()
    return tf.keras.models.load_model(PATH_TO_STORE_MODEL + FILE_NAME)


def download_model():
    """Download facenet h5 file from google drive
    """

    if not Path(PATH_TO_STORE_MODEL + FILE_NAME).exists():
        print("Downloading facenet model...")

        # Make directory to store downloaded model
        Path(PATH_TO_STORE_MODEL).mkdir(
            parents=True, exist_ok=True,
        )

        # Download from google drives
        download_file_from_google_drive(FILE_ID, PATH_TO_STORE_MODEL + FILE_NAME)
