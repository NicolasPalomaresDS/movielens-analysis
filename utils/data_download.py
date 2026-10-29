import hashlib
import urllib.request
import zipfile
from pathlib import Path

BASE_URL = "https://files.grouplens.org/datasets/movielens"
ZIP_NAME = "ml-32m.zip"
RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def md5sum(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.md5()

    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)

    return h.hexdigest()


def download(url: str, dest: Path) -> None:
    last_pct = -1

    def progress(blocks: int, block_size: int, total: int) -> None:
        nonlocal last_pct

        if total <= 0:
            return

        pct = int(min(100, blocks * block_size * 100 / total))

        if pct != last_pct:
            last_pct = pct

            print(
                f"\rDownloading {dest.name}: {pct:3d}%",
                end="",
                flush=True
            )

    urllib.request.urlretrieve(url, dest, reporthook=progress)
    print()


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    extracted = RAW_DIR / "ml-32m"
    zip_path = RAW_DIR / ZIP_NAME

    if (extracted / "ratings.csv").exists():
        print("Dataset already in", extracted)
        return

    if not zip_path.exists():
        download(f"{BASE_URL}/{ZIP_NAME}", zip_path)

    with urllib.request.urlopen(f"{BASE_URL}/{ZIP_NAME}.md5") as r:
        expected = r.read().decode().split()[0]

    actual = md5sum(zip_path)

    if actual != expected:
        zip_path.unlink()

        raise RuntimeError(
            f"Invalid checksum ({actual} != {expected}). "
            "File deleted."
        )

    print("Checksum OK. Extracting...")

    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(RAW_DIR)

    zip_path.unlink()
    print("Ready:", extracted)


if __name__ == "__main__":
    main()