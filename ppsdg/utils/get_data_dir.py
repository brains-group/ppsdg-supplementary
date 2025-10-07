import os
from pathlib import Path


def get_data_dir() -> Path:
    if "DATA_DIR" not in os.environ:
        print("DATA_DIR is not set in .env. We will default to ./.data")
        data_dir = Path(".data")
    else:
        data_dir = Path(os.environ.get("DATA_DIR"))
    return data_dir
