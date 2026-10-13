"""
Sort remodel photos by technical quality.

Usage:
    python sort_photos.py

Required libraries install automatically on first run, and the script
asks you for the photos folder.

Creates <photos folder>_sorted/ with:
    good/        sharp, well exposed
    maybe/       borderline, give these a quick look
    rejected/    blurry or badly exposed
    duplicates/  near-copies of a better photo
    report.csv   scores for every photo, so you can tune thresholds
"""
import csv
import importlib
import shutil
import subprocess
import sys
from pathlib import Path

# (import name, pip package name)
REQUIREMENTS = [
    ("PIL", "pillow"),
    ("cv2", "opencv-python"),
    ("numpy", "numpy"),
    ("imagehash", "imagehash"),
]


def ensure_requirements():
    missing = []
    for module, package in REQUIREMENTS:
        try:
            importlib.import_module(module)
        except ImportError:
            missing.append(package)
    if missing:
        print(f"Installing required libraries: {', '.join(missing)}")
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", *missing]
            )
        except subprocess.CalledProcessError:
            sys.exit(
                "\nCouldn't install the libraries automatically. "
                f"Try running: pip install {' '.join(missing)}"
            )
        importlib.invalidate_caches()
        print("Installation complete.\n")


ensure_requirements()

import cv2
import imagehash
import numpy as np
from PIL import Image, ImageOps

# ---- Thresholds (tune using report.csv) ----
BLUR_REJECT = 40       # sharpness below this = rejected
BLUR_MAYBE = 90        # sharpness below this = maybe
DARK_REJECT = 45       # average brightness below this = rejected
DARK_MAYBE = 75
BRIGHT_REJECT = 215    # average brightness above this = rejected
BRIGHT_MAYBE = 190
CLIPPED_MAYBE = 0.30   # fraction of pure black or pure white pixels
DUPLICATE_DISTANCE = 6 # lower = stricter duplicate matching
ANALYSIS_WIDTH = 1000  # resize before scoring so results are comparable

EXTENSIONS = {".webp", ".jpg", ".jpeg", ".png"}


def analyze(path):
    img = Image.open(path)
    img = ImageOps.exif_transpose(img).convert("RGB")
    phash = imagehash.phash(img)

    w, h = img.size
    scale = ANALYSIS_WIDTH / w
    small = img.resize((ANALYSIS_WIDTH, max(1, int(h * scale))))
    gray = cv2.cvtColor(np.array(small), cv2.COLOR_RGB2GRAY)

    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
    brightness = float(gray.mean())
    clipped = float(((gray < 8) | (gray > 247)).mean())
    return sharpness, brightness, clipped, phash, (w, h)


def classify(sharp, bright, clipped):
    reasons = []
    status = "good"

    if sharp < BLUR_REJECT:
        status = "rejected"; reasons.append("blurry")
    elif sharp < BLUR_MAYBE:
        status = "maybe"; reasons.append("slightly soft")

    if bright < DARK_REJECT:
        status = "rejected"; reasons.append("too dark")
    elif bright < DARK_MAYBE and status == "good":
        status = "maybe"; reasons.append("dark")

    if bright > BRIGHT_REJECT:
        status = "rejected"; reasons.append("too bright")
    elif bright > BRIGHT_MAYBE and status == "good":
        status = "maybe"; reasons.append("bright")

    if clipped > CLIPPED_MAYBE and status == "good":
        status = "maybe"; reasons.append("blown highlights/crushed shadows")

    return status, "; ".join(reasons)


def main():
    while True:
        raw = input("Enter the path to your photos folder: ").strip()
        raw = raw.strip('"').strip("'")  # handles drag-and-drop / copied paths
        src = Path(raw).expanduser()
        if src.is_dir():
            break
        print(f"  Folder not found: {src}\n  Please try again.\n")

    out = src.parent / f"{src.name}_sorted"
    for name in ("good", "maybe", "rejected", "duplicates"):
        (out / name).mkdir(parents=True, exist_ok=True)

    files = sorted(p for p in src.rglob("*") if p.suffix.lower() in EXTENSIONS)
    print(f"Found {len(files)} photos. Analyzing...")

    rows = []
    for i, p in enumerate(files, 1):
        try:
            sharp, bright, clipped, ph, size = analyze(p)
        except Exception as e:
            print(f"  Skipped {p.name}: {e}")
            continue
        status, why = classify(sharp, bright, clipped)
        rows.append({"path": p, "sharp": sharp, "bright": bright,
                     "clipped": clipped, "hash": ph, "status": status,
                     "reason": why, "size": size})
        if i % 50 == 0:
            print(f"  {i}/{len(files)}")

    # Duplicate detection among photos that weren't rejected.
    # Sharpest photo in each group stays; the rest become duplicates.
    candidates = sorted((r for r in rows if r["status"] != "rejected"),
                        key=lambda r: r["sharp"], reverse=True)
    kept = []
    for r in candidates:
        if any(r["hash"] - k["hash"] <= DUPLICATE_DISTANCE for k in kept):
            r["status"] = "duplicates"
            r["reason"] = (r["reason"] + "; " if r["reason"] else "") + "near-duplicate"
        else:
            kept.append(r)

    # Copy files and write report. Prefix avoids name collisions.
    with open(out / "report.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["file", "status", "reason", "sharpness",
                     "brightness", "clipped_pct", "width", "height"])
        for r in rows:
            dest = out / r["status"] / r["path"].name
            n = 1
            while dest.exists():
                dest = out / r["status"] / f"{r['path'].stem}_{n}{r['path'].suffix}"
                n += 1
            shutil.copy2(r["path"], dest)
            wr.writerow([r["path"].name, r["status"], r["reason"],
                         round(r["sharp"], 1), round(r["bright"], 1),
                         round(r["clipped"] * 100, 1), *r["size"]])

    counts = {}
    for r in rows:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    print("\nDone.")
    for k in ("good", "maybe", "rejected", "duplicates"):
        print(f"  {k}: {counts.get(k, 0)}")
    print(f"\nResults in: {out}")


if __name__ == "__main__":
    main()
