import argparse
import tarfile
import urllib.request
import zipfile
from pathlib import Path
import kagglehub
import numpy as np
from scipy.ndimage import gaussian_filter
from aether.io.image import save_image

def generate_synthetic_nebula(width: int = 512, height: int = 512, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[-1:1:complex(0, height), -1:1:complex(0, width)]
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)

    ring = np.exp(-((r - 0.45) ** 2) / 0.02)
    inner_bubble = np.exp(-(r**2) / 0.08) * 0.7
    outer_halo = np.exp(-(r**2) / 0.35) * 0.4

    noise1 = rng.normal(0, 1, (height, width))
    noise2 = rng.normal(0, 1, (height, width))
    turb1 = gaussian_filter(noise1, sigma=width * 0.04)
    turb2 = gaussian_filter(noise2, sigma=width * 0.015)
    turb1 = (turb1 - turb1.min()) / (turb1.max() - turb1.min())
    turb2 = (turb2 - turb2.min()) / (turb2.max() - turb2.min())

    gas = (ring * 1.2 + inner_bubble + outer_halo) * (0.6 + 0.4 * turb1) * (0.8 + 0.2 * turb2)
    spiral = 0.2 * np.sin(4 * theta + 3 * r) * np.exp(-r)
    gas = np.maximum(0.0, gas + spiral)

    red = gas * 0.95 + 0.15 * ring
    green = gas * 0.35 + 0.5 * inner_bubble
    blue = gas * 0.2 + 0.8 * inner_bubble

    num_stars = 180
    star_y = rng.integers(0, height, num_stars)
    star_x = rng.integers(0, width, num_stars)
    star_bright = rng.uniform(0.6, 1.0, num_stars)

    rgb = np.stack([red, green, blue], axis=-1)
    for sy, sx, sb in zip(star_y, star_x, star_bright):
        rgb[sy, sx] += sb
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                ny, nx = sy + dy, sx + dx
                if 0 <= ny < height and 0 <= nx < width:
                    rgb[ny, nx] += sb * 0.25

    return np.clip(rgb, 0.0, 1.0).astype(np.float32)

def download_and_extract(url: str | None, kaggle_name: str | None, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    filename = (url.split("/")[-1] if url else kaggle_name.split("/")[-1]) or "dataset.zip"
    filepath = dest_dir / filename
    if url:
        urllib.request.urlretrieve(url, filepath)
    else:
        kagglehub.dataset_download(kaggle_name, output_dir=str(filepath))

    if filepath.name.endswith(".zip"):
        with zipfile.ZipFile(filepath, "r") as zip_ref:
            zip_ref.extractall(dest_dir)
    elif filepath.name.endswith((".tar.gz", ".tgz")):
        with tarfile.open(filepath, "r:gz") as tar_ref:
            tar_ref.extractall(dest_dir)

def main() -> None:
    parser = argparse.ArgumentParser(description="Aether Dataset Downloader and Generator")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--url", type=str, help="URL of archive to download")
    group.add_argument("--kagglehub", type=str, help="Dataset identifier from Kaggle")
    group.add_argument("--generate-sample", action="store_true", help="Generate synthetic nebula sample")
    parser.add_argument("--dest", type=str, default="data/dataset", help="Destination folder")
    parser.add_argument("--sample-output", type=str, default="data/samples/synthetic_nebula.png")
    args = parser.parse_args()

    if args.generate_sample:
        arr = generate_synthetic_nebula()
        save_image(args.sample_output, arr)
    else:
        dest_path = Path.cwd() / args.dest
        download_and_extract(args.url, args.kagglehub, dest_path)

if __name__ == "__main__":
    main()