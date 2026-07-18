"""
features_lib.py
Shared feature extraction for the LIVE simulation (stage 7).

Rather than re-implement the feature maths here (which would drift from the
pipelines the moment either side is edited), we DELEGATE to the very functions
that produced image_features.csv / audio_features.csv:

    stage 4  ->  <image pipeline>.normalise(), .features()
    stage 5  ->  <audio pipeline>.normalise(), .features()
"""
from importlib import import_module
from pathlib import Path

import numpy as np
from PIL import Image

from config import IMG_SIZE, SR

# locate the pipeline modules by filename (digit-prefixed) 
_SRC = Path(__file__).resolve().parent


def _load_pipeline(glob_pattern):
    """Import the single src module matching e.g. '*image_pipeline.py' by stem."""
    matches = sorted(_SRC.glob(glob_pattern))
    if not matches:
        raise ImportError(f"No pipeline module matching {glob_pattern} in {_SRC}")
    return import_module(matches[0].stem)


_img = _load_pipeline("*image_pipeline.py")
_aud = _load_pipeline("*audio_pipeline.py")


# IMAGE 
def image_features_from_path(path):
    """Real photo on disk -> feature vector, via the stage-4 normalise+features."""
    img, _ = _img.normalise(path)
    return _img.features(img).astype(np.float32)


def image_features(pil_or_array):
    """Already-loaded image (PIL.Image or HxWx3 array) -> feature vector.

    Applies the SAME canonicalisation as training: face-crop -> resize ->
    autocontrast. Accepts an array (e.g. an OpenCV BGR frame) or a PIL image.
    """
    if isinstance(pil_or_array, np.ndarray):
        arr = pil_or_array
        if arr.ndim == 3 and arr.shape[2] == 3:
            arr = arr[:, :, ::-1] # BGR (OpenCV) -> RGB
        img = Image.fromarray(arr.astype(np.uint8)).convert("RGB")
    else:
        img = pil_or_array.convert("RGB")

    box = _img.detect_face_box(img)
    if box is not None:
        img = img.crop(box)
    else:
        from PIL import ImageOps
        side = min(img.size)
        img = ImageOps.fit(img, (side, side), method=Image.LANCZOS)
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    from PIL import ImageOps
    img = ImageOps.autocontrast(img, cutoff=1)
    return _img.features(img).astype(np.float32)


# AUDIO 
def audio_features_from_path(path):
    """Real recording on disk -> feature vector, via the stage-5 normalise+features."""
    y, _ = _aud.normalise(path)
    return _aud.features(y).astype(np.float32)


def audio_features(y):
    """Already-loaded mono waveform (float array at SR) -> feature vector.

    Applies the same fixed-duration / normalisation that training used by
    routing through the stage-5 feature function. If you have a RAW clip at an
    arbitrary sample rate, prefer audio_features_from_path so it is resampled and
    trimmed identically to training.
    """
    y = np.asarray(y, dtype=np.float32)
    return _aud.features(y).astype(np.float32)


# re-export the canonical constants so callers import them from one place
__all__ = ["image_features", "image_features_from_path",
           "audio_features", "audio_features_from_path", "SR", "IMG_SIZE"]