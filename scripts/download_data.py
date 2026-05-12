"""Download and unpack TLG datasets.

This script downloads:
  - QVHighlights annotations (JSONL) + official pre-extracted features
    (SlowFast + CLIP video, CLIP text, PANN audio).
  - Charades-STA annotations (TXT).

For Charades-STA, the feature releases are hosted by various authors and
URLs drift; the script prints clear instructions if a feature archive is
missing instead of guessing.

Usage:
    python scripts/download_data.py --dataset qvhighlights --out data/qvhighlights
    python scripts/download_data.py --dataset charades   --out data/charades_sta

Notes
-----
QVHighlights pre-extracted features are large (~30 GB). If you only want to
verify the pipeline, pass `--annotations-only`.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import tarfile
import urllib.request
import zipfile
from typing import Iterable

# ---------------------------------------------------------------------------
# URL registry
# ---------------------------------------------------------------------------
#
# QVHighlights:
#   The dataset is released by Lei et al. (2021) and hosted on the
#   Moment-DETR GitHub. The canonical download links live in
#   https://github.com/jayleicn/moment_detr/blob/main/data/README.md .
#
# Charades-STA:
#   Annotations originate from Gao et al. (2017). The TXT files are mirrored
#   in several places; we use the maintained mirror in the 2D-TAN repo.

QVHIGHLIGHTS = {
    "annotations": [
        # The annotations bundle (~1 MB) includes train/val/test jsonls.
        ("https://nlp.cs.unc.edu/data/jielei/qvh/data.tar.gz",
         "annotations.tar.gz"),
    ],
    "features": [
        # SlowFast+CLIP video features and CLIP text features (~30 GB total).
        ("https://nlp.cs.unc.edu/data/jielei/qvh/qvhighlights_features.tar.gz",
         "features.tar.gz"),
    ],
    "pann_audio": [
        # PANN audio features (~3 GB) released alongside UMT.
        ("https://github.com/TencentARC/UMT/releases/download/v1.0/"
         "qvhighlights_pann_features.tar.gz",
         "pann_features.tar.gz"),
    ],
}

CHARADES_STA = {
    "annotations": [
        ("https://raw.githubusercontent.com/jiyanggao/TALL/master/"
         "exp_data/Charades/charades_sta_train.txt",
         "charades_sta_train.txt"),
        ("https://raw.githubusercontent.com/jiyanggao/TALL/master/"
         "exp_data/Charades/charades_sta_test.txt",
         "charades_sta_test.txt"),
    ],
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _download(url: str, dest: str) -> None:
    if os.path.exists(dest):
        print(f"[skip] {dest} already exists")
        return
    print(f"[download] {url}\n  -> {dest}")
    tmp = dest + ".part"
    try:
        with urllib.request.urlopen(url) as resp, open(tmp, "wb") as out:
            total = int(resp.headers.get("Content-Length", "0"))
            done = 0
            chunk = 1 << 20
            while True:
                buf = resp.read(chunk)
                if not buf:
                    break
                out.write(buf)
                done += len(buf)
                if total:
                    pct = 100.0 * done / total
                    sys.stdout.write(f"\r  {pct:5.1f}% ({done >> 20} MB)")
                    sys.stdout.flush()
            sys.stdout.write("\n")
        shutil.move(tmp, dest)
    except Exception as e:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise RuntimeError(f"download failed for {url}: {e}") from e


def _extract(archive: str, dest_dir: str) -> None:
    print(f"[extract] {archive} -> {dest_dir}")
    os.makedirs(dest_dir, exist_ok=True)
    if archive.endswith(".tar.gz") or archive.endswith(".tgz"):
        with tarfile.open(archive, "r:gz") as tf:
            tf.extractall(dest_dir)
    elif archive.endswith(".zip"):
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(dest_dir)
    else:
        print(f"[warn] don't know how to extract {archive}; leaving as-is")


def _run(items: Iterable[tuple[str, str]], out_dir: str) -> list[str]:
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for url, name in items:
        dest = os.path.join(out_dir, name)
        _download(url, dest)
        paths.append(dest)
    return paths


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


def download_qvhighlights(out: str, annotations_only: bool, skip_audio: bool) -> None:
    os.makedirs(out, exist_ok=True)

    # 1. Annotations
    ann_paths = _run(QVHIGHLIGHTS["annotations"], out)
    for p in ann_paths:
        _extract(p, os.path.join(out, "annotations"))

    if annotations_only:
        print("[done] annotations only — skipping features.")
        return

    # 2. Video + text features
    feat_paths = _run(QVHIGHLIGHTS["features"], out)
    for p in feat_paths:
        _extract(p, os.path.join(out, "features"))

    # 3. Audio features (optional)
    if not skip_audio:
        try:
            audio_paths = _run(QVHIGHLIGHTS["pann_audio"], out)
            for p in audio_paths:
                _extract(p, os.path.join(out, "features"))
        except RuntimeError as e:
            print(f"[warn] PANN audio download failed ({e}). You can train "
                  f"with use_audio=False, or fetch the archive manually from "
                  f"the UMT release page and unpack it under "
                  f"{os.path.join(out, 'features', 'pann_features')}.")


def download_charades(out: str) -> None:
    os.makedirs(out, exist_ok=True)
    _run(CHARADES_STA["annotations"], out)
    print(
        "\n[done] Charades-STA annotations downloaded.\n"
        "Charades-STA feature releases vary by author — the official "
        "Charades distribution hosts the raw videos (https://prior.allenai.org/projects/charades), "
        "from which VGG/optical-flow features can be re-extracted. The TLG\n"
        "community most commonly uses the features released by Zhang et al.\n"
        "(2020b) for 2D-TAN. Place them under:\n"
        f"    {out}/vgg_features/{{vid}}.npy\n"
        f"    {out}/optical_flow/{{vid}}.npy\n"
        f"    {out}/glove_features/qid{{qid}}.npy\n"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset",
                   choices=["qvhighlights", "charades"], required=True)
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument("--annotations-only", action="store_true")
    p.add_argument("--skip-audio", action="store_true",
                   help="(QVHighlights) skip the PANN audio features bundle")
    args = p.parse_args()

    if args.dataset == "qvhighlights":
        download_qvhighlights(args.out, args.annotations_only, args.skip_audio)
    else:
        download_charades(args.out)


if __name__ == "__main__":
    main()
