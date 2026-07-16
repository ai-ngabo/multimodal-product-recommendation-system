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
