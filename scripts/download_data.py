"""
Download Phase 1 datasets via Kaggle API.
Requires: kaggle CLI configured (~/.kaggle/kaggle.json)

Run:
  python scripts/download_data.py
  python scripts/download_data.py --datasets scada_pipeline
  python scripts/download_data.py --phase 1
"""
from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.dataset_registry import load_registry

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]


def download_kaggle(dataset_key: str, meta: dict) -> bool:
    kaggle_path = meta.get("kaggle_path")
    if not kaggle_path:
        logger.warning("No kaggle_path for %s", dataset_key)
        return False

    raw_dir = ROOT / meta["local_raw"]
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Skip if already downloaded
    existing_csvs = list(raw_dir.glob("*.csv"))
    if existing_csvs:
        logger.info("Already downloaded: %s (%s)", dataset_key, existing_csvs[0].name)
        return True

    logger.info("Downloading %s from kaggle: %s", dataset_key, kaggle_path)
    cmd = ["kaggle", "datasets", "download", "-d", kaggle_path, "-p", str(raw_dir), "--unzip"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error("Kaggle download failed for %s:\n%s", dataset_key, result.stderr)
        return False

    csvs = list(raw_dir.glob("*.csv"))
    logger.info("Downloaded %s: %d CSV file(s) in %s", dataset_key, len(csvs), raw_dir)
    return len(csvs) > 0


def download_uci(dataset_key: str, meta: dict) -> bool:
    import urllib.request

    url = meta.get("uci_url")
    if not url:
        logger.warning("No uci_url for %s", dataset_key)
        return False

    raw_dir = ROOT / meta["local_raw"]
    raw_dir.mkdir(parents=True, exist_ok=True)

    zip_path = raw_dir / "uci_hydraulic.zip"
    if list(raw_dir.glob("*.txt")) or list(raw_dir.glob("*.dat")):
        logger.info("Already downloaded: %s", dataset_key)
        return True

    logger.info("Downloading UCI dataset: %s", url)
    try:
        urllib.request.urlretrieve(url, zip_path)
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(raw_dir)
        zip_path.unlink(missing_ok=True)
        logger.info("Downloaded and extracted %s to %s", dataset_key, raw_dir)
        return True
    except Exception as e:
        logger.error("UCI download failed for %s: %s", dataset_key, e)
        return False


def prepare_manual_dataset(dataset_key: str, meta: dict) -> bool:
    raw_dir = ROOT / meta["local_raw"]
    raw_dir.mkdir(parents=True, exist_ok=True)

    existing_files = [path for path in raw_dir.rglob("*") if path.is_file()]
    if existing_files:
        logger.info("Already present: %s (%d file(s))", dataset_key, len(existing_files))
        return True

    source_url = meta.get("dataset_url") or meta.get("repo_url") or meta.get("uci_url")
    logger.warning(
        "Dataset %s requires manual download. Place the raw files under %s and rerun normalization. Source: %s",
        dataset_key,
        raw_dir,
        source_url or "unknown",
    )
    return False


def main():
    parser = argparse.ArgumentParser(description="Download datasets")
    parser.add_argument("--datasets", nargs="+", default=None, help="Dataset keys to download")
    parser.add_argument("--phase", type=int, default=1, help="Download all datasets at this phase")
    args = parser.parse_args()

    registry = load_registry()

    if args.datasets:
        targets = {k: registry[k] for k in args.datasets if k in registry}
    else:
        targets = {k: v for k, v in registry.items() if v.get("phase", 99) <= args.phase}

    if not targets:
        logger.error("No matching datasets found.")
        sys.exit(1)

    results = {}
    for key, meta in targets.items():
        source = meta.get("source", "kaggle")
        if source == "kaggle":
            ok = download_kaggle(key, meta)
        elif source == "uci":
            ok = download_uci(key, meta)
        elif source in {"mendeley", "github", "phmsa"}:
            ok = prepare_manual_dataset(key, meta)
        else:
            logger.warning("Unknown source '%s' for %s", source, key)
            ok = False
        results[key] = ok

    print("\n=== DOWNLOAD SUMMARY ===")
    for key, ok in results.items():
        status = "OK" if ok else "FAILED"
        print(f"  {key}: {status}")

    failed = [k for k, ok in results.items() if not ok]
    if failed:
        print(f"\nFailed: {failed}")
        print("See DOWNLOAD_DATA.md for manual instructions.")
        sys.exit(1)


if __name__ == "__main__":
    main()
