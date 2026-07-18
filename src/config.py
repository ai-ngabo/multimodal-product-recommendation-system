# Shared configuration

from pathlib import Path
import numpy as np

# repoducibility
SEED = 43
np.random.seed(SEED)

# Paths
ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "unrefined" / "datasets"
PROC = ROOT / "data" / "cleaned"
PLOTS = ROOT / "outputs" / "plots"
FEATURES = ROOT / "outputs" / "features"
MODELS = ROOT / "outputs" / "models"
RAW_IMAGES  = ROOT / "data" / "unrefined" / "images"
FACES = ROOT / "data" / "cleaned" / "faces"
RAW_AUDIO = ROOT / "data" / "unrefined" / "audios"
CLEAN_AUDIO = ROOT / "data" / "cleaned" / "audios"

# Biometric Identity Management variables (FACE + VOICE)
USERS = ["alain", "hakim", "joella", "sonia"]
FACE_EXPRESSION = ["neutral", "smiling", "surprised"]
PHRASES = {"phrase1": "Yes, approve", "phrase2": "Confirm transaction"} 


# Image settings
IMG_SIZE = 128

# Voice Settings
SR       = 16000
DURATION = 2.0