import os
import zipfile
from pathlib import Path
import requests


def download_data(url:str ,folder_name :str , is_zip:bool):
  data_path = Path("data/")
  folder_path = data_path / folder_name
  if folder_path.is_dir():
      print(f"folder already exists")
  else : 
          folder_path.mkdir(parents=True , exist_ok=True)
          path = data_path/f"{folder_name}.zip" if is_zip else folder_path
          with open(path , "wb") as f :
            request = requests.get(url=url , verify=False)
            f.write(request.content)
          if is_zip:
           with open(path,"r")as zip_ref:
              zip_ref.extractall(folder_path)    
           os.remove(path)    
  return folder_path    



import shutil

def download_data_GPT(
    url: str,
    folder_name: str,
    is_zip: bool,
    data_path: Path = Path("data/"),
) -> Path:
    """Downloads data from a given URL and optionally extracts ZIP files.

    Args:
        url (str): URL to download the data from.
        folder_name (str): Name of the folder to store the data.
        is_zip (bool): Whether the file to download is a ZIP file.
        data_path (Path): Path where data folders will be created.

    Returns:
        Path: Path to the downloaded/extracted folder.
    """
    folder_path = data_path / folder_name
    # Create the folder if it doesn't exist
    folder_path.mkdir(parents=True, exist_ok=True)
    
    # Check if folder already exists
    if any(folder_path.iterdir()):  # Check if folder contains files
        print(f"Folder '{folder_path}' already exists.")
        return folder_path

    # File download
    print(f"Downloading from {url} to {folder_path}...")
    file_path = data_path / f"{folder_name}.zip" if is_zip else folder_path / f"{folder_name}.data"

    try:
        response = requests.get(url, stream=True, timeout=30 , verify=False)  # Use streaming for large files
        response.raise_for_status()  # Raise an error for failed requests
        with open(file_path, "wb") as f:
            shutil.copyfileobj(response.raw, f)

        # If ZIP, extract it
        if is_zip:
            print(f"Extracting {file_path}...")
            shutil.unpack_archive(file_path, folder_path)
            file_path.unlink()  # Delete ZIP file after extraction
    except Exception as e:
        print(f"Error during download or extraction: {e}")
        raise

    print(f"Data available at {folder_path}")
    return folder_path
