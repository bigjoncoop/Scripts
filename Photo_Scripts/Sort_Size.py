
"""
move_small_photos.py

Moves low-resolution photos out of a folder so you can review or delete them.

What it does:
  1. Installs its dependencies automatically if they're missing
     (Pillow, plus the optional pillow-avif-plugin for AVIF support).
  2. Prompts for the folder containing your photos.
  3. Prompts for the minimum size in pixels (press Enter for the 1600 default).
  4. Moves any photo whose longest side is smaller than that size into a
     "too small" subfolder inside the chosen folder.

Notes:
  - Only the top level of the folder is scanned (subfolders are ignored).
  - Existing files in "too small" are never overwritten; duplicates are skipped.
  - Supported formats: JPG, JPEG, PNG, WEBP, BMP, TIF, TIFF, GIF, AVIF.

Usage:
  python move_small_photos.py
"""

import importlib
import subprocess
import sys
from pathlib import Path

DEFAULT_MIN_SIZE = 1600


def ensure_package(import_name, pip_name=None, required=True):
    """Import a package, installing it with pip first if it's missing."""
    pip_name = pip_name or import_name
    try:
        return importlib.import_module(import_name)
    except ImportError:
        print(f"Installing {pip_name}...")
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", pip_name]
            )
            importlib.invalidate_caches()
            return importlib.import_module(import_name)
        except Exception as error:
            if required:
                print(f"Could not install {pip_name}: {error}")
                sys.exit(1)
            print(f"Optional package {pip_name} not installed: {error}")
            return None


# Install dependencies if needed
ensure_package("PIL", "Pillow")
# Optional: adds AVIF support on older Pillow versions
ensure_package("pillow_avif", "pillow-avif-plugin", required=False)

import shutil  # noqa: E402
from PIL import Image  # noqa: E402


def ask_for_folder():
    while True:
        answer = input("Enter the folder containing the photos: ").strip()
        answer = answer.strip('"').strip("'")  # handles drag-and-drop quotes
        folder = Path(answer).expanduser()
        if folder.is_dir():
            return folder
        print(f"'{answer}' is not a valid folder. Try again.\n")


def ask_for_size():
    while True:
        answer = input(
            f"Minimum size in pixels for the longest side "
            f"[{DEFAULT_MIN_SIZE}]: "
        ).strip()
        if not answer:
            return DEFAULT_MIN_SIZE
        if answer.isdigit() and int(answer) > 0:
            return int(answer)
        print("Please enter a whole number greater than 0.\n")


folder = ask_for_folder()
minimum_size = ask_for_size()
destination = folder / "too small"

destination.mkdir(exist_ok=True)

extensions = {
    ".jpg", ".jpeg", ".png", ".webp",
    ".bmp", ".tif", ".tiff", ".gif", ".avif"
}

moved = 0

for photo in folder.iterdir():
    if not photo.is_file() or photo.suffix.lower() not in extensions:
        continue

    try:
        with Image.open(photo) as image:
            width, height = image.size

        if max(width, height) < minimum_size:
            target = destination / photo.name

            # Avoid overwriting files with the same name
            if target.exists():
                print(f"Skipped duplicate: {photo.name}")
                continue

            shutil.move(str(photo), str(target))
            print(f"Moved: {photo.name} ({width}x{height})")
            moved += 1

    except Exception as error:
        print(f"Could not process {photo.name}: {error}")

print(f"\nDone. Moved {moved} photos under {minimum_size}px to '{destination}'.")

