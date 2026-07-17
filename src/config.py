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

# Biometric Identity Management variables
USERS = ["alain", "hakim", "joella", "sonia"]
FACE_EXPRESSION = ["neutral", "smiling", "surprised"]
PHRASES = {"phrase1": "Yes, approve", "phrase2": "Confirm transaction"} 

# Image settings
IMG_SIZE = 128