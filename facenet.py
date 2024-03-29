# Speical thank to Hiroki Taniai for providing pretrained facenet model.
# https://github.com/nyoki-mtl/keras-facenet

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from pathlib import Path
from downloaders import download_file_from_google_drive
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Static variables
FILE_ID = "1iWWmn8eTHo6AK-l3rBQB-YfzMbPa-GXX"
PATH_TO_STORE_MODEL = "./models/"
FILE_NAME = "facenet.h5"


def load_model():
    """Load pretrained facenet model
    """
    print("Preparing facenet model...")
    download_model()
    return tf.keras.models.load_model(PATH_TO_STORE_MODEL + FILE_NAME)


def download_model():
    """Download facenet h5 file from google drive
    """

    if not Path(PATH_TO_STORE_MODEL + FILE_NAME).exists():
        print("Downloading", FILE_NAME, "...")

        # Make directory to store downloaded model
        Path(PATH_TO_STORE_MODEL).mkdir(
            parents=True, exist_ok=True,
        )

        # Download from google drives
        download_file_from_google_drive(FILE_ID, PATH_TO_STORE_MODEL + FILE_NAME)
