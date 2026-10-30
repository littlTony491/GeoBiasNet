"""Download and install the H3.6M exponential-map TXT dataset."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import shutil
import stat
import tempfile
from urllib.parse import urlparse
import zipfile


ROOT = Path(__file__).resolve().parent
# Maintained mirror linked by the LearnTrajDep authors:
SOURCE_PAGE = "https://github.com/wei-mao-2019/LearnTrajDep#Get-the-data"
DEFAULT_URL = "https://drive.google.com/uc?id=1FbBXEQlLgD7Me6JFm_2MJbGf9T72d3u_"
SUBJECTS = ("S1", "S5", "S6", "S7", "S8", "S9", "S11")
ACTIONS = (
    "walking", "eating", "smoking", "discussion", "directions", "greeting",
    "phoning", "posing", "purchases", "sitting", "sittingdown", "takingphoto",
    "waiting", "walkingdog", "walkingtogether",
)
EXPECTED_FILES = tuple(f"{subject}/{action}_{part}.txt"
                       for subject in SUBJECTS for action in ACTIONS for part in (1, 2))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_dataset(directory):
    directory = Path(directory).resolve()
    missing = [name for name in EXPECTED_FILES if not (directory/name).is_file()]
    if missing:
        raise ValueError(f"missing {len(missing)} H3.6M files in {directory}; examples: {', '.join(missing[:4])}")
    for name in EXPECTED_FILES:
        with (directory/name).open(encoding="utf-8-sig") as handle:
            row = handle.readline().strip().split(",")
        try:
            valid = len(row) == 99 and all(math.isfinite(float(value)) for value in row)
        except ValueError:
            valid = False
        if not valid:
            raise ValueError(f"expected 99 finite exponential-map values per row: {directory/name}")
    return {"directory": str(directory), "subjects": len(SUBJECTS), "actions": len(ACTIONS),
            "files": len(EXPECTED_FILES), "columns": 99}


def archive_members(archive):
    expected = {name.casefold(): name for name in EXPECTED_FILES}
    selected = {}
    for member in archive.infolist():
        name = PurePosixPath(member.filename.replace("\\", "/"))
        if name.is_absolute() or ".." in name.parts or any(":" in part for part in name.parts):
            raise ValueError(f"unsafe archive path: {member.filename}")
        if stat.S_ISLNK(member.external_attr >> 16):
            raise ValueError(f"archive links are not supported: {member.filename}")
        if member.is_dir() or len(name.parts) < 2:
            continue
        relative = "/".join(name.parts[-2:]).casefold()
        if relative in expected:
            canonical = expected[relative]
            if canonical in selected:
                raise ValueError(f"duplicate dataset file in archive: {canonical}")
            selected[canonical] = member
    missing = sorted(set(EXPECTED_FILES) - selected.keys())
    if missing:
        raise ValueError(f"archive is missing {len(missing)} required files; examples: {', '.join(missing[:4])}")
    return selected


def install_archive(archive_path, output_dir, source=None):
    archive_path = Path(archive_path).resolve()
    output = Path(output_dir).resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"output is not empty; check it or choose another --output-dir: {output}")
    with zipfile.ZipFile(archive_path) as archive:
        members = archive_members(archive)
        output.parent.mkdir(parents=True, exist_ok=True)
        # Stage beside the destination so installation is an atomic rename.
        with tempfile.TemporaryDirectory(prefix=".h36m-", dir=output.parent) as temporary:
            staging = Path(temporary).resolve()
            if staging.parent != output.parent or not staging.name.startswith(".h36m-"):
                raise RuntimeError("unexpected staging directory")
            for relative, member in members.items():
                target = staging/relative  # Relative paths come only from EXPECTED_FILES.
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member) as src, target.open("wb") as dst:
                    shutil.copyfileobj(src, dst)  # Reading to EOF also checks each member's CRC.
            summary = validate_dataset(staging)
            manifest = {"source": source, "archive_sha256": sha256(archive_path),
                        "installed_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                        "subjects": list(SUBJECTS), "actions": list(ACTIONS),
                        "files": summary["files"], "columns": 99}
            (staging/"download_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8")
            if output.exists():
                output.rmdir()  # Empty directories only; existing data is never removed.
            staging.replace(output)
    return validate_dataset(output)


def download_archive(url, cache_dir, proxy=None):
    if urlparse(url).scheme not in ("https", "http"):
        raise ValueError("--url must be an HTTP or HTTPS URL; use --archive for a local ZIP")
    cache = Path(cache_dir).resolve()
    cache.mkdir(parents=True, exist_ok=True)
    path = cache/("h36m-" + hashlib.sha256(url.encode()).hexdigest()[:12] + ".zip")
    if path.is_file():
        if not zipfile.is_zipfile(path):
            raise ValueError(f"invalid cached ZIP; move it aside before retrying: {path}")
        print(f"Using cached archive: {path}", flush=True)
        return path
    try:
        import gdown
    except ImportError as error:
        raise RuntimeError("install download dependencies first: python -m pip install -r requirements-download.txt") from error
    print(f"Downloading H3.6M to {path}", flush=True)
    result = gdown.download(url=url, output=str(path), proxy=proxy, quiet=False,
                           resume=True, use_cookies=False, timeout=30, retries=2)
    if result is None or not zipfile.is_zipfile(path):
        raise RuntimeError("download did not produce a ZIP; retry or use --archive with a downloaded H3.6M ZIP")
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--url", default=None, help="alternative public ZIP URL")
    source.add_argument("--archive", type=Path, help="install an already downloaded ZIP without network access")
    parser.add_argument("--output-dir", type=Path, default=ROOT/"dataset"/"h3.6m")
    parser.add_argument("--cache-dir", type=Path, default=ROOT/"dataset"/".downloads")
    parser.add_argument("--proxy", default=None, help="optional HTTP/SOCKS proxy for the download")
    parser.add_argument("--check-only", action="store_true", help="validate existing data without downloading")
    args = parser.parse_args(argv)
    try:
        if args.output_dir.exists():
            try:
                summary = validate_dataset(args.output_dir)
            except (ValueError, OSError):
                if args.check_only:
                    raise
                if not args.output_dir.is_dir() or any(args.output_dir.iterdir()):
                    raise FileExistsError(f"existing data is incomplete or has another format; choose a new --output-dir: {args.output_dir}")
            else:
                print("Dataset already available: " + json.dumps(summary), flush=True)
                return 0
        elif args.check_only:
            raise FileNotFoundError(f"dataset directory does not exist: {args.output_dir}")
        url = args.url or DEFAULT_URL
        archive = args.archive if args.archive is not None else download_archive(url, args.cache_dir, args.proxy)
        summary = install_archive(archive, args.output_dir, source=str(args.archive) if args.archive else url)
    except Exception as error:
        parser.exit(1, f"{type(error).__name__}: {error}\n")
    print("H3.6M ready: " + json.dumps(summary), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
