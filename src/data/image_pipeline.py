"""
image_pipeline.py

Task 2: Image Data Collection and Processing — automated pipeline.

Loads each team member's facial images (neutral, smiling, surprised),
validates naming/completeness, detects and crops faces, applies
augmentations (rotation, flip, grayscale), extracts color histogram +
HOG embedding features, and saves everything to image_features.csv.

Usage:
    python src/data/image_pipeline.py
    python src/data/image_pipeline.py --images-dir data/unrefined/images --output data/cleaned/image_features.csv

This module is import-safe: other scripts (e.g. the face recognition
model in src/models/) can `from src.data.image_pipeline import build_feature_dataframe`
and reuse the same feature extraction logic instead of duplicating it.
"""

import os
import argparse

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from skimage.feature import hog

EXPRESSIONS = ["neutral", "smiling", "surprised"]

AUGMENTATIONS = {
    "original": lambda im: im,
    "rotated": lambda im: _augment_rotate(im),
    "flipped": lambda im: _augment_flip(im),
    "grayscale": lambda im: _augment_grayscale(im),
}

_face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


# --------------------------------------------------------------------------- #
# Step 1: Validation
# --------------------------------------------------------------------------- #
def check_completeness(images_dir: str) -> pd.DataFrame:
    """Verify every member folder has exactly neutral/smiling/surprised.jpg."""
    members = sorted(
        d for d in os.listdir(images_dir)
        if os.path.isdir(os.path.join(images_dir, d))
    )
    report = []
    for member in members:
        member_dir = os.path.join(images_dir, member)
        found = set(os.listdir(member_dir))
        expected = {f"{e}.jpg" for e in EXPRESSIONS}
        missing = expected - found
        report.append({
            "member": member,
            "missing": ", ".join(sorted(missing)) if missing else "none",
            "status": "OK" if not missing else "INCOMPLETE",
        })
    return pd.DataFrame(report)


# --------------------------------------------------------------------------- #
# Step 2: Face detection
# --------------------------------------------------------------------------- #
def detect_and_crop_face(pil_img: Image.Image):
    """Detect the largest face in the image; fall back to the full image."""
    img_cv = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    faces = _face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
    )
    if len(faces) == 0:
        return img_cv, False
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    return img_cv[y:y + h, x:x + w], True


# --------------------------------------------------------------------------- #
# Step 3: Augmentations
# --------------------------------------------------------------------------- #
def _augment_rotate(img_cv, angle: int = 15):
    h, w = img_cv.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(img_cv, M, (w, h), borderMode=cv2.BORDER_REPLICATE)


def _augment_flip(img_cv):
    return cv2.flip(img_cv, 1)


def _augment_grayscale(img_cv):
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


# --------------------------------------------------------------------------- #
# Step 4: Feature extraction
# --------------------------------------------------------------------------- #
def extract_color_histogram(img_cv, bins: int = 8):
    hist_features = []
    for ch in range(3):
        hist = cv2.calcHist([img_cv], [ch], None, [bins], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        hist_features.extend(hist.tolist())
    return hist_features


def extract_hog_embedding(img_cv, size: int = 64):
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (size, size))
    return hog(
        resized, orientations=8, pixels_per_cell=(16, 16),
        cells_per_block=(1, 1), feature_vector=True
    ).tolist()


# --------------------------------------------------------------------------- #
# Step 5: Full pipeline
# --------------------------------------------------------------------------- #
def build_feature_dataframe(images_dir: str) -> pd.DataFrame:
    """Run the full pipeline (detect -> augment -> extract) over every image."""
    members = sorted(
        d for d in os.listdir(images_dir)
        if os.path.isdir(os.path.join(images_dir, d))
    )
    rows = []
    for member in members:
        for expr in EXPRESSIONS:
            path = os.path.join(images_dir, member, f"{expr}.jpg")
            if not os.path.exists(path):
                print(f"⚠️  Skipping missing file: {path}")
                continue

            pil_img = Image.open(path)
            face_crop, face_found = detect_and_crop_face(pil_img)

            for aug_name, aug_fn in AUGMENTATIONS.items():
                aug_img = aug_fn(face_crop)
                hist_feats = extract_color_histogram(aug_img)
                hog_feats = extract_hog_embedding(aug_img)

                row = {
                    "member": member,
                    "expression": expr,
                    "augmentation": aug_name,
                    "face_detected": face_found,
                    "source_path": path,
                }
                row.update({f"hist_{i}": v for i, v in enumerate(hist_feats)})
                row.update({f"hog_{i}": v for i, v in enumerate(hog_feats)})
                rows.append(row)

    return pd.DataFrame(rows)


def run_pipeline(images_dir: str, output_path: str) -> pd.DataFrame:
    print(f"Checking image completeness in: {images_dir}")
    report = check_completeness(images_dir)
    print(report.to_string(index=False))

    incomplete = report[report["status"] == "INCOMPLETE"]
    if len(incomplete) > 0:
        print(f"\n⚠️  {len(incomplete)} member(s) incomplete — continuing anyway.")

    print("\nRunning face detection, augmentation, and feature extraction...")
    df = build_feature_dataframe(images_dir)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\n✅ Saved {len(df)} rows to {output_path}")
    return df


# --------------------------------------------------------------------------- #
# CLI entry point
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description="Image data preprocessing pipeline")
    parser.add_argument(
        "--images-dir", default="data/unrefined/images",
        help="Root folder containing one subfolder per team member"
    )
    parser.add_argument(
        "--output", default="data/cleaned/image_features.csv",
        help="Where to save the extracted features CSV"
    )
    args = parser.parse_args()
    run_pipeline(args.images_dir, args.output)


if __name__ == "__main__":
    main()
