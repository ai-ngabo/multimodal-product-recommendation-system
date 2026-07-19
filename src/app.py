#Mini App
"""Running the entire pipeline end-to-end: EDA -> merge -> product model ->
image -> audio -> biometric models -> simulation."""
import subprocess, sys
from pathlib import Path

SRC = Path(__file__).resolve().parent
STAGES = [
    "1_eda.py", "2_clean_merge_data.py", "3_product_model.py",
    "4_image_pipeline.py", "5_audio_pipeline.py",
    "6_biometric_models.py", "7_simulation.py",
]

def main():
    for s in STAGES:
        print(f"\n{'='*70}\nRUNNING {s}\n{'='*70}")
        r = subprocess.run([sys.executable, str(SRC / s)])
        if r.returncode != 0:
            print(f"[FAIL] {s} exited with {r.returncode}")
            sys.exit(r.returncode)
    print("\n[ALL STAGES COMPLETE]")

if __name__ == "__main__":
    main()
