import urllib.request
import zipfile
import tarfile
import argparse
import kagglehub
from pathlib import Path

def download_and_extract(url: str | None, kaggle_name: str | None, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    filename = (url.split('/')[-1] if url else kaggle_name.split('/')[-1]) or 'dataset.zip'
    filepath = dest_dir / filename
    if url:
        print(f"Downloading {url} to {filepath}...")
        urllib.request.urlretrieve(url, filepath)
    else:
        print(f"Downloading {kaggle_name} from kagglehub to {filepath}...")
        path = kagglehub.dataset_download(kaggle_name, output_dir=filepath)    
    print("Download complete.")
    print(f"Extracting {filepath}...")
    if filepath.name.endswith('.zip'):
        with zipfile.ZipFile(filepath, 'r') as zip_ref:
            zip_ref.extractall(dest_dir)
    elif filepath.name.endswith('.tar.gz') or filepath.name.endswith('.tgz'):
        with tarfile.open(filepath, 'r:gz') as tar_ref:
            tar_ref.extractall(dest_dir)
    else:
        print("Unknown archive format. Skipping extraction.")
    print(f"Dataset successfully installed in {dest_dir}")

def main() -> None:
    parser = argparse.ArgumentParser(description="Aether Dataset Installer")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--url", type=str, help="URL of the dataset archive to download")
    group.add_argument("--kagglehub", type=str, help="Name of dataset from kagglehub")
    parser.add_argument("--dest", type=str, default="data/dataset", help="Destination folder inside 'data/'")
    args = parser.parse_args()
    dest_path = Path.cwd() / args.dest
    download_and_extract(args.url, args.kagglehub, dest_path)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}")