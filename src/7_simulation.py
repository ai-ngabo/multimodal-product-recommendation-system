"""
7_simulation.py  -  Multimodal Authentication + Product Recommendation demo
==========================================================================

MULTIMODAL LOGIC (explicit AND-gate):
    authorised = face_known AND voice_known AND (face_id == voice_id)
  Each modality must (a) clear its rejection threshold AND (b) not be flagged as
  an out-of-distribution impostor by a novelty (distance-to-centroid) gate.

Features come from features_lib, which delegates to the stage-4/5 pipeline
functions, so a live probe is featurised EXACTLY as the training CSVs were.

Usage:
    python 7_simulation.py - demo
    python 7_simulation.py - face path/to/face.jpg - voice path/to/voice.wav
    python 7_simulation.py - member hakim    # authorised run for one member
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import librosa

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config import (FEATURES, MODELS, FACES, CLEAN_AUDIO, RAW_IMAGES, RAW_AUDIO,
                    USERS, SR, MERGED_CSV)
from features_lib import (image_features, image_features_from_path,
                          audio_features_from_path)

# enrollment: link each biometric identity to a CRM customer_id 
# Any member -> customer_id mapping works.
ENROLLMENT = dict(zip(USERS, [100, 101, 102, 103]))
FACE_NOVELTY_FACTOR = 1.15
VOICE_NOVELTY_FACTOR = 1.10

C = dict(GRN="\033[92m", RED="\033[91m", YEL="\033[93m", CYN="\033[96m",
         B="\033[1m", X="\033[0m")
def cprint(msg, col=""): print(f"{C.get(col, '')}{msg}{C['X']}")

# load assets
face = joblib.load(MODELS / "face_model.joblib")
voice = joblib.load(MODELS / "voice_model.joblib")
product = joblib.load(MODELS / "product_model.joblib")
img_feat_df = pd.read_csv(FEATURES / "image_features.csv")
aud_feat_df = pd.read_csv(FEATURES / "audio_features.csv")
merged = pd.read_csv(MERGED_CSV)


def _centroids(df, feat_cols):
    """Per-member centroid + max intra-class radius, for the novelty gate."""
    g = df.groupby("member")[feat_cols].mean()
    radii = {}
    for m in g.index:
        pts = df.loc[df.member == m, feat_cols].values
        radii[m] = float(np.linalg.norm(pts - g.loc[m].values, axis=1).max())
    return g, radii


face_cent, face_rad = _centroids(img_feat_df, face["feat_cols"])
voice_cent, voice_rad = _centroids(aud_feat_df, voice["feat_cols"])


def _novelty_ok(vec, centroids, radii, factor):
    """True if vec falls within factor x the nearest member's intra-class radius."""
    d = np.linalg.norm(centroids.values - vec, axis=1)
    j = int(np.argmin(d))
    nearest = centroids.index[j]
    return d[j] <= radii[nearest] * factor, nearest, float(d[j])


def _identify(vec, bundle, cent, rad, factor):
    proba = bundle["model"].predict_proba(vec.reshape(1, -1))[0]
    idx = int(np.argmax(proba))
    conf = float(proba[idx])
    member = bundle["label_encoder"].classes_[idx]
    ok_conf = conf >= bundle["reject_threshold"]
    ok_nov, _, dist = _novelty_ok(vec, cent, rad, factor)
    return dict(member=member, conf=conf, known=(ok_conf and ok_nov), dist=dist)


def identify_face(img_or_path):
    vec = (image_features_from_path(img_or_path)
           if isinstance(img_or_path, (str, Path))
           else image_features(img_or_path))
    return _identify(vec, face, face_cent, face_rad, FACE_NOVELTY_FACTOR)


def identify_voice(path):
    vec = audio_features_from_path(path)
    return _identify(vec, voice, voice_cent, voice_rad, VOICE_NOVELTY_FACTOR)


def recommend_product(customer_id):
    row = merged[merged.customer_id == customer_id]
    if row.empty:
        base = merged.median(numeric_only=True)
        ctx = {c: base.get(c, 0) for c in product["numeric"]}
        for c in product["categ"]:
            ctx[c] = "Unknown"
    else:
        r = row.iloc[0]
        ctx = {c: r[c] for c in product["numeric"]}
        for c in product["categ"]:
            ctx[c] = r[c]
    X = pd.DataFrame([ctx])[product["numeric"] + product["categ"]]
    proba = product["model"].predict_proba(X)[0]
    idx = int(np.argmax(proba))
    return product["label_encoder"].classes_[idx], float(proba[idx])


def run_transaction(face_input, voice_path, label):
    cprint(f"\n{'=' * 60}", "CYN")
    cprint(f" TRANSACTION ATTEMPT -- {label}", "B")
    cprint(f"{'=' * 60}", "CYN")

    cprint("\n[1/3] Facial Recognition ...", "B")
    f = identify_face(face_input)
    print(f"      predicted: {f['member']:<10} conf={f['conf']:.2f} "
          f"dist={f['dist']:.1f}  -> {'KNOWN' if f['known'] else 'UNKNOWN'}")
    if not f["known"]:
        cprint("      ACCESS DENIED - face not recognised.", "RED")
        return

    cprint("\n[2/3] Voiceprint Verification ...", "B")
    v = identify_voice(voice_path)
    print(f"      predicted: {v['member']:<10} conf={v['conf']:.2f} "
          f"dist={v['dist']:.1f}  -> {'KNOWN' if v['known'] else 'UNKNOWN'}")
    if not v["known"]:
        cprint("      ACCESS DENIED - voice not recognised.", "RED")
        return

    if f["member"] != v["member"]:
        cprint(f"\n      ACCESS DENIED - identity mismatch "
               f"(face={f['member']} vs voice={v['member']}).", "RED")
        return

    cprint(f"\n      + Multimodal match confirmed: {f['member']}", "GRN")
    cid = ENROLLMENT.get(f["member"])
    cprint(f"\n[3/3] Product Recommendation (customer_id={cid}) ...", "B")
    prod, p = recommend_product(cid)
    cprint(f"      >>> RECOMMENDED PRODUCT: {prod}  (model confidence {p:.2f})", "GRN")
    cprint("      TRANSACTION APPROVED.", "GRN")


# demo helpers
# The demo uses the held-out capture per member (the expression/phrase that the
# leave-one-out evaluation tested on), so it exercises genuinely unseen inputs.
DEMO_FACE_EXPR = "surprised"
DEMO_VOICE_PHRASE = "phrase2"     # matches PHRASES keys in config


def _face_probe(member):
    """Path to a member's held-out face capture (canonical copy from stage 4)."""
    p = FACES / f"{member}_{DEMO_FACE_EXPR}.png"
    return p if p.exists() else next(RAW_IMAGES.glob(f"{member}_{DEMO_FACE_EXPR}.*"))


def _voice_probe(member):
    """Path to a member's held-out voice capture (canonical copy from stage 5)."""
    p = CLEAN_AUDIO / f"{member}_{DEMO_VOICE_PHRASE}.wav"
    return p if p.exists() else next(RAW_AUDIO.glob(f"{member}_*"))


def _impostor_voice_path():
    """Synthesise a buzzy out-of-distribution clip and write it to a temp wav."""
    import soundfile as sf
    rng = np.random.default_rng(999)
    t = np.linspace(0, 1.5, int(SR * 1.5), endpoint=False)
    y = 0.6 * np.sign(np.sin(2 * np.pi * 300 * t)) + 0.2 * rng.standard_normal(len(t))
    y = (y / np.max(np.abs(y))).astype(np.float32)
    tmp = MODELS.parent / "_impostor_voice.wav"
    sf.write(tmp, y, SR)
    return tmp


def _impostor_face_array():
    """A random noise image - unlike any enrolled face, should fail novelty."""
    rng = np.random.default_rng(999)
    return rng.integers(0, 255, (128, 128, 3), dtype=np.uint8)


def main():
    ap = argparse.ArgumentParser(description="Multimodal auth + product recommendation")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--face")
    ap.add_argument("--voice")
    ap.add_argument("--member", default=USERS[0],
                    help=f"member for a single authorised run: {USERS}")
    args = ap.parse_args()

    def _resolve(p):
        p = Path(p)
        return p if p.is_absolute() else (ROOT / p)

    if args.face and args.voice:
        face_p, voice_p = _resolve(args.face), _resolve(args.voice)
        run_transaction(str(face_p), str(voice_p),
                        f"file: {face_p.name} + {voice_p.name}")
        return

    if not args.demo:
        m = args.member
        run_transaction(_face_probe(m), _voice_probe(m), f"AUTHORISED USER ({m})")
        return

    cprint("\n########  MULTIMODAL AUTHENTICATION SYSTEM - DEMO  ########", "B")
    m0, m1 = USERS[0], USERS[1]

    # 1. authorised: member's own held-out face + voice
    run_transaction(_face_probe(m0), _voice_probe(m0), f"AUTHORISED USER ({m0})")

    # 2. impostor: novel face + buzzy voice -> rejected at the face novelty gate
    run_transaction(_impostor_face_array(), _impostor_voice_path(),
                    "UNAUTHORISED ATTEMPT (impostor face + voice)")

    # 3. mismatch: one member's face + another's voice -> identities disagree
    run_transaction(_face_probe(m0), _voice_probe(m1),
                    f"MISMATCH ATTEMPT ({m0} face + {m1} voice)")

    cprint("\n########  END OF DEMO  ########\n", "B")


if __name__ == "__main__":
    main()
