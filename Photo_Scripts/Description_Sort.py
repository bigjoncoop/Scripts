"""
sort_photos.py

Sorts photos into Bathroom, Kitchen, or Other-Review folders using an AI
image classifier (OpenAI's CLIP model, run locally on your CPU).

What it does:
  1. Installs its dependencies automatically if they're missing
     (Pillow, transformers, torch).
  2. Prompts for the folder containing your photos.
  3. Prompts for the minimum confidence (0 to 1) needed to trust the AI's
     pick (press Enter for the 0.45 default).
  4. Classifies each photo as a bathroom, a kitchen, or something else.
  5. COPIES each photo into a "Sorted_Photos" folder created next to the
     folder you chose, inside Bathroom, Kitchen, or Other-Review. Your
     originals are never moved or changed.

Notes:
  - Photos where the AI's confidence is below the threshold go to
    Other-Review so you can check them by hand.
  - Only the top level of the folder is scanned (subfolders are ignored).
  - Supported formats: JPG, JPEG, PNG, WEBP, BMP.
  - The first run downloads the CLIP model (a few hundred MB) and the torch
    install is large, so it can take a while. Later runs are much faster.

Usage:
  python sort_photos.py
"""

import importlib
import subprocess
import sys
from pathlib import Path

DEFAULT_THRESHOLD = 0.45


def ensure_package(import_name, pip_name=None):
    """Import a package, installing it with pip first if it's missing."""
    pip_name = pip_name or import_name
    try:
        return importlib.import_module(import_name)
    except ImportError:
        print(f"Installing {pip_name}... (this may take a few minutes)")
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", pip_name]
            )
            importlib.invalidate_caches()
            return importlib.import_module(import_name)
        except Exception as error:
            print(f"Could not install {pip_name}: {error}")
            sys.exit(1)


# Install dependencies if needed
ensure_package("PIL", "Pillow")
ensure_package("torch", "torch")
ensure_package("transformers", "transformers")

from shutil import copy2  # noqa: E402
from PIL import Image  # noqa: E402
from transformers import pipeline  # noqa: E402


def ask_for_folder():
    while True:
        answer = input("Enter photo folder path: ").strip()
        answer = answer.strip('"').strip("'")  # handles drag-and-drop quotes
        folder = Path(answer).expanduser()
        if folder.is_dir():
            return folder
        print(f"'{answer}' is not a valid folder. Try again.\n")


def ask_for_threshold():
    while True:
        answer = input(
            f"Minimum confidence from 0 to 1 [{DEFAULT_THRESHOLD}]: "
        ).strip()
        if not answer:
            return DEFAULT_THRESHOLD
        try:
            value = float(answer)
            if 0 <= value <= 1:
                return value
        except ValueError:
            pass
        print("Please enter a number between 0 and 1.\n")


SOURCE = ask_for_folder()
THRESHOLD = ask_for_threshold()
OUTPUT = SOURCE.parent / "Sorted_Photos"

print("\nLoading AI model (first run downloads it)...")
classifier = pipeline(
    "zero-shot-image-classification",
    model="openai/clip-vit-base-patch32",
    device=-1
)

labels = [
    "a photo of a bathroom interior",
    "a photo of a kitchen interior",
    "a photo unrelated to a kitchen or bathroom"
]

extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
for folder in ["Bathroom", "Kitchen", "Other-Review"]:
    (OUTPUT / folder).mkdir(parents=True, exist_ok=True)

for path in SOURCE.iterdir():
    if not path.is_file() or path.suffix.lower() not in extensions:
        continue
    try:
        with Image.open(path) as im:
            image = im.convert("RGB")
        result = classifier(image, candidate_labels=labels)
        best = result[0]
        folder = (
            "Bathroom" if best["label"] == labels[0] else
            "Kitchen" if best["label"] == labels[1] else
            "Other-Review"
        )
        if best["score"] < THRESHOLD:
            folder = "Other-Review"
        copy2(path, OUTPUT / folder / path.name)
        print(folder, path.name, round(best["score"], 2))
    except Exception as e:
        print("Skipped:", path.name, e)

print("Finished:", OUTPUT)
