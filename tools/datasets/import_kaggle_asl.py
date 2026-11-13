"""
Import the Google "Isolated Sign Language Recognition" dataset (250 ASL signs,
MediaPipe landmarks, recorded by 21 signers) into Mudra's ASL pack.

Get it from Kaggle (free account, accept the competition rules, and check
that they allow your use): https://www.kaggle.com/competitions/asl-signs/data
You need train.csv and the train_landmark_files folder (tens of GB unzipped).

Setup (once):
    python -m pip install pandas pyarrow

Run:
    python tools/datasets/import_kaggle_asl.py --data "D:/asl-signs" --top 50 --train

Each sequence becomes one repetition, and each signer is one "recording", so
the evaluation tests on signers the model has never seen.

Notes:
  * The dataset stores image coordinates scaled to 0..1, without the video
    size. --aspect sets width/height (default 1.0); try 0.5625 (9:16) if the
    trained model does poorly on your live signing.
  * If left and right hands seem swapped in practice, re-import with
    --swap-hands into a fresh pack.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from common import label_from_folder, open_pack, pick_labels, train_moving

FPS = 30.0
POSE_POINTS = (0, 11, 12)  # nose, left shoulder, right shoulder


def sequence_to_frames(df, aspect: float = 1.0, swap_hands: bool = False, scale: float = 1000.0) -> list[dict]:
    """One parquet file (frame, type, landmark_index, x, y, z) -> Mudra clip frames."""
    import numpy as np
    w, h = scale * aspect, scale
    frames = []
    for frame_no, g in df.groupby("frame", sort=True):
        hands = []
        for kind, handed in (("right_hand", "Right"), ("left_hand", "Left")):
            part = g[g["type"] == kind].sort_values("landmark_index")
            if len(part) != 21:
                continue
            xyz = part[["x", "y", "z"]].to_numpy(dtype=float)
            if not np.all(np.isfinite(xyz)):
                continue
            if swap_hands:
                handed = "Left" if handed == "Right" else "Right"
            lm = np.column_stack([xyz[:, 0] * w, xyz[:, 1] * h, xyz[:, 2] * w])
            hands.append({"handed": handed, "lm": lm.tolist()})
        pose = None
        pp = g[g["type"] == "pose"].set_index("landmark_index")
        if all(i in pp.index for i in POSE_POINTS):
            pts = pp.loc[list(POSE_POINTS), ["x", "y", "z"]].to_numpy(dtype=float)
            if np.all(np.isfinite(pts[:, :2])):
                pose = [[p[0] * w, p[1] * h, p[2] * w] for p in pts]
        frames.append({"t": round(float(frame_no) / FPS, 4), "hands": hands, "pose": pose})
    t0 = frames[0]["t"] if frames else 0.0
    for f in frames:
        f["t"] = round(f["t"] - t0, 4)
    return frames


def main(argv=None):
    ap = argparse.ArgumentParser(description="Import Kaggle ASL Signs landmarks into a Mudra pack")
    ap.add_argument("--data", required=True, help="folder with train.csv and train_landmark_files/")
    ap.add_argument("--pack", default="asl")
    ap.add_argument("--top", type=int, help="only the N most common signs (e.g. 50)")
    ap.add_argument("--only", help='only these signs, comma separated, e.g. "hello,thankyou"')
    ap.add_argument("--max-per-sign", type=int, default=0, help="at most this many sequences per sign")
    ap.add_argument("--aspect", type=float, default=1.0, help="video width / height (default 1.0)")
    ap.add_argument("--swap-hands", action="store_true", help="treat left_hand as right and vice versa")
    ap.add_argument("--train", action="store_true", help="train the moving-sign model when done")
    args = ap.parse_args(argv)

    try:
        import pandas as pd
    except ImportError:
        sys.exit("pandas isn't installed. Run: python -m pip install pandas pyarrow")
    root = Path(args.data)
    csv = root / "train.csv"
    if not csv.exists():
        sys.exit(f"train.csv not found in {root}")
    index = pd.read_csv(csv)
    index["label"] = index["sign"].map(label_from_folder)
    index = index.dropna(subset=["label"])
    labels = pick_labels(index["label"].value_counts().to_dict(), args.only, args.top)
    index = index[index["label"].isin(labels)]
    if args.max_per_sign:
        index = index.groupby("label", group_keys=False).head(args.max_per_sign)
    if index.empty:
        sys.exit("None of the requested signs were found in train.csv.")

    pack = open_pack(args.pack)
    have_seq = {r.get("source") for r in pack.data.load() if r.get("source")}
    print(f"{len(labels)} signs, {len(index)} sequences into {pack.meta.get('name')}.")
    added = skipped = 0
    for n, row in enumerate(index.itertuples(index=False), 1):
        source = f"kaggle-asl:{row.sequence_id}"
        if source in have_seq:
            skipped += 1
            continue
        try:
            df = pd.read_parquet(root / row.path, columns=["frame", "type", "landmark_index", "x", "y", "z"])
            frames = sequence_to_frames(df, args.aspect, args.swap_hands)
        except Exception as exc:  # a damaged file shouldn't stop the import
            print(f"\n  {row.path}: {exc}")
            continue
        session = f"signer-{row.participant_id}"
        if pack.sign_kind(row.label) == "still":
            continue
        if pack.data.add_clips(row.label, session, [frames], source=source):
            pack.set_sign_kind(row.label, "moving")
            added += 1
        if n % 25 == 0:
            print(f"\r  {n}/{len(index)}", end="", flush=True)
    print(f"\nImported {added} sequences ({skipped} were already there).")
    if args.train:
        train_moving(pack)
    else:
        print("Open Mudra, choose this pack, and press Train model in the Train tab.")


if __name__ == "__main__":
    main()
