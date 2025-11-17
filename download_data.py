import os
import zipfile
import requests
from pathlib import Path


def download_rd_tablebench():
    url = "https://huggingface.co/datasets/reducto/rd-tablebench/resolve/main/rd-tablebench.zip"
    zip_file = "rd-tablebench.zip"
    data_dir = "data"

    Path(data_dir).mkdir(exist_ok=True)

    # Download
    print("Downloading rd-tablebench.zip...")
    response = requests.get(url)
    with open(zip_file, "wb") as f:
        f.write(response.content)

    # Extract
    print("Extracting to data/...")
    with zipfile.ZipFile(zip_file, "r") as zip_ref:
        zip_ref.extractall(data_dir)

    os.remove(zip_file)
    print("Done")


def main():
    print("Downloading rd-tablebench...")
    download_rd_tablebench()


if __name__ == "__main__":
    main()
