"""Stage 4: Image data collection & processing.

    member photos are read from data/unrefined/images/, normalised to a
    canonical form (EXIF-corrected -> face-cropped -> square -> IMG_SIZE ->
    illumination-equalised), cached to data/cleaned/faces/, then augmented (more than 2 per
    image) and reduced to a feature vector.

Steps: ingest -> normalise -> display grid -> augmentations/image -> features(csv)
Features: 24-bin RGB colour histogram + 64-dim low-res grayscale embedding
        + HOG structure descriptor -> outputs/features/image_features.csv
"""
import numpy as np
from PIL import Image, ImageOps, ImageEnhance
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import hashlib
from skimage.feature import hog
from config import (RAW_IMAGES, PLOTS, FEATURES, USERS, FACE_EXPRESSION, 
                    IMG_SIZE, SEED, ROOT, RAW_IMAGES, FACES)

rng = np.random.default_rng(SEED)

IMG_FORMAT = (".jpg", ".jpeg", ".png", ".webp", ".heic", ".bmp")
META_COLS = ["member", "expression", "variant", "source"]

def find_source(user, expression):
    """Locate the raw capture for (user, expression). Supports:
         data/unrefined/images/<user>/<expression>.jpg     (nested)
         data/unrefined/images/<user>_<expression>.jpg     (flat)
    """
    stems = [f"{user}_{expression}", expression]
    roots = [RAW_IMAGES / user, RAW_IMAGES]
    for root in roots:
        if not root.is_dir():
            continue
        for f in root.iterdir():
            if f.suffix.lower() in IMG_FORMAT and f.stem.lower() in stems:
                return f
    raise FileNotFoundError(
        f"No Image Found for {user}/{expression}."
    )

# Normalizing photos taken by phone
try:
    import cv2

    def _load(name):
        c = cv2.CascadeClassifier(cv2.data.haarcascades + name)
        return None if c.empty() else c

    _CASCADES = [c for c in (_load("haarcascade_frontalface_default.xml"),
                             _load("haarcascade_frontalface_alt2.xml"),
                             _load("haarcascade_profileface.xml")) if c]
except Exception as ex:
    print("[CHECK] cascade unavailable:", ex)
    _CASCADES = []

def normalise(path):
    """Raw capture -> canonical IMG_SIZE x IMG_SIZE RGB face crop."""
    img = Image.open(path)
    img = ImageOps.exif_transpose(img) #rotation
    img = img.convert("RGB")

    box = detect_face_box(img)  #cropping
    if box is not None:
        img = img.crop(box)
    else:
        img = ImageOps.fit(img, (min(img.size), min(img.size)))  # centre-crop

    img = img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)   #scaling
    img = ImageOps.autocontrast(img, cutoff=1)  #lighting
    return img, box is not None


def detect_face_box(img, margin=0.35):
    """Largest detected face, expanded by `margin` and squared. None if no cascade."""
    if not _CASCADES:
        return None
    gray = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2GRAY)
    faces = []
    for casc in _CASCADES:
        faces = casc.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3,
                                      minSize=(60, 60))
        if len(faces):
            break
    if len(faces) == 0:
        return None

    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    cx, cy = x + w / 2, y + h / 2
    half = max(w, h) * (1 + margin) / 2
    W, H = img.size
    return (max(0, int(cx - half)), max(0, int(cy - half)),
            min(W, int(cx + half)), min(H, int(cy + half)))

# describe features per user's face
def features(img):
    """24-bin RGB histogram + 8x8 grayscale embedding + HOG structure."""
    arr = np.asarray(img.convert("RGB"))
    hist = []
    for ch in range(3):
        h, _ = np.histogram(arr[:, :, ch], bins=8, range=(0, 256), density=True)
        hist.extend(h)
    gray = np.asarray(img.convert("L"))
    emb = np.asarray(Image.fromarray(gray).resize((8, 8))).flatten() / 255.0
    hog_vec = hog(gray, orientations=9, pixels_per_cell=(16, 16),
                  cells_per_block=(2, 2), block_norm="L2-Hys", feature_vector=True)
    return np.concatenate([hist, emb, hog_vec])

# Data Augmentation
def _seed_from(key):
    return int(hashlib.md5(key.encode()).hexdigest()[:8], 16)

def augment(img, key):
    """ augmentations, deterministic per source image."""
    r = np.random.default_rng(_seed_from(key) ^ SEED)
    return {
        "rot":    img.rotate(int(r.integers(-15, 16)), resample=Image.BILINEAR),
        "flip":   img.transpose(Image.FLIP_LEFT_RIGHT),
        "gray":   img.convert("L").convert("RGB"),
        "bright": ImageEnhance.Brightness(img).enhance(float(r.uniform(0.7, 1.3))),
    }

# CLI entry point
def main():
    rows, grid_imgs = [], []
 
    for m in USERS:
        for e in FACE_EXPRESSION:
            src = find_source(m, e)
            img, face_found = normalise(src)
            img.save(FACES / f"{m}_{e}.png")      # canonical copy
            grid_imgs.append((f"{m}\n{e}", img, face_found))
 
            variants = {"orig": img, **augment(img, f"{m}_{e}")}
            for vname, vimg in variants.items():
                feats = features(vimg)
                rows.append({"member": m, "expression": e, "variant": vname,
                             "source": src.name,
                             **{f"f{i}": v for i, v in enumerate(feats)}})
 
    # display sample grid (doubles as detection QC) 
    fig, axes = plt.subplots(len(USERS), len(FACE_EXPRESSION), figsize=(10, 13))
    for ax, (title, img, ok) in zip(axes.ravel(), grid_imgs):
        ax.imshow(img)
        ax.set_title(title, fontsize=9, color="black" if ok else "red")
        ax.set_xticks([]); ax.set_yticks([])
        if not ok:
            for s in ax.spines.values():
                s.set_visible(True); s.set_color("red"); s.set_linewidth(3)
        else:
            ax.axis("off")
    fig.suptitle("Sample faces — 4 users × 3 expressions (real captures)\n"
                 "red = face detection failed, centre-crop fallback used",
                 y=1.02, fontsize=13)
    fig.tight_layout()
    fig.savefig(PLOTS / "image_pipeline" / "image_samples.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
 
    # display augmentations for one image 
    demo, _ = normalise(find_source(USERS[0], "smiling"))
    augs = {"original": demo, **augment(demo, f"{USERS[0]}_smiling")}
    fig, axes = plt.subplots(1, len(augs), figsize=(15, 4))
    for ax, (name, im) in zip(axes, augs.items()):
        ax.imshow(im); ax.set_title(name); ax.axis("off")
    fig.suptitle(f"Image augmentations ({USERS[0]} / smiling)", fontsize=13)
    fig.tight_layout()
    fig.savefig(PLOTS / "image_pipeline" /"image_augmentations.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
 
    # savinf image features
    df = pd.DataFrame(rows)
    df.to_csv(FEATURES / "image_features.csv", index=False)
    df.to_csv(ROOT / "image_features.csv", index=False) 
 
    # QC summary (the image-pipeline analogue of merge validation)
    n_expected = len(USERS) * len(FACE_EXPRESSION)
    n_detected = sum(ok for _, _, ok in grid_imgs)
    per_member = df.groupby("member").size().to_dict()
 
    print(f"[Done] image_features.csv -> {df.shape[0]} rows, {df.shape[1]} cols "
          f"({df.shape[1] - len(META_COLS)} feature dims)")
    print(f"[CHECK] base images loaded : {len(grid_imgs)}/{n_expected}")
    print(f"[CHECK] face detected      : {n_detected}/{len(grid_imgs)} "
          f"({n_detected / len(grid_imgs):.0%}) — rest used centre-crop fallback")
    print(f"[CHECK] rows per member    : {per_member}")
    if not _CASCADES:
        print("[CHECK] WARNING: OpenCV cascade unavailable — every image was "
              "centre-cropped. Install opencv-python for face detection.")
 
    assert len(grid_imgs) == n_expected, "missing captures — see errors above"
    assert df.groupby("member").size().nunique() == 1, \
        f"class imbalance across members: {per_member}"
    print("[Done] Quality Control assertions passed.")
 
 
if __name__ == "__main__":
    main()