#!/usr/bin/env python3
"""
Convert a Roboflow YOLO export (images + bbox labels + data.yaml) into the grouped
ground-truth JSON that evaluate_real.py consumes.

Each photo's cards are grouped into player1 / community / player2 by clustering the
label bounding boxes by vertical (y) position into 3 rows -- top -> player1,
middle -> community, bottom -> player2 (the app's layout). The card identity comes
from the YOLO class id via data.yaml's class map. This gives an exact ground truth
straight from your existing annotations -- no manual re-labelling.

Usage (from the backend/ directory, with the project venv):
    python eval/labels_to_ground_truth.py \
        --dataset eval/photos/final_poker_dataset/final_poker_dataset \
        --out eval/ground_truth.json
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import yaml

IMG_EXTS = ('.jpg', '.jpeg', '.png')


def load_class_names(data_yaml):
    data = yaml.safe_load(Path(data_yaml).read_text(encoding='utf-8'))
    names = data['names']
    if isinstance(names, dict):
        return {int(k): v for k, v in names.items()}
    return {i: n for i, n in enumerate(names)}


def cluster_rows(cards):
    """cards: list of (name, y_center, x_center). Split into 3 rows (top->bottom)
    at the two largest vertical gaps."""
    cards = sorted(cards, key=lambda c: c[1])
    if len(cards) <= 3:
        return [cards]
    ys = [c[1] for c in cards]
    gaps = sorted(range(len(ys) - 1), key=lambda i: ys[i + 1] - ys[i], reverse=True)
    boundaries = sorted(gaps[:2])
    rows, start = [], 0
    for b in boundaries:
        rows.append(cards[start:b + 1])
        start = b + 1
    rows.append(cards[start:])
    return rows


def convert(dataset_dir):
    ds = Path(dataset_dir)
    names = load_class_names(ds / 'data.yaml')
    img_dir, lbl_dir = ds / 'images' / 'train', ds / 'labels' / 'train'
    ground_truth, warnings = {}, []

    for lbl in sorted(lbl_dir.glob('*.txt')):
        image = next((img_dir / (lbl.stem + ext) for ext in IMG_EXTS
                      if (img_dir / (lbl.stem + ext)).exists()), None)
        if image is None:
            warnings.append(f"{lbl.name}: no matching image")
            continue

        cards = []
        for line in lbl.read_text().splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            cid, xc, yc = int(float(parts[0])), float(parts[1]), float(parts[2])
            cards.append((names.get(cid, f"?{cid}"), yc, xc))

        rows = cluster_rows(cards)
        if len(rows) != 3:
            warnings.append(f"{image.name}: clustered into {len(rows)} row(s) "
                            f"({len(cards)} cards) -- needs a manual check")
            continue

        entry = {
            'player1': [c[0] for c in rows[0]],
            'community': [c[0] for c in rows[1]],
            'player2': [c[0] for c in rows[2]],
        }
        ground_truth[image.name] = entry
        if len(entry['player1']) != 2 or len(entry['player2']) != 2 \
                or not (3 <= len(entry['community']) <= 5):
            warnings.append(
                f"{image.name}: unusual row sizes "
                f"p1={len(entry['player1'])} comm={len(entry['community'])} "
                f"p2={len(entry['player2'])}"
            )
    return ground_truth, warnings


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dataset', required=True, help="Roboflow export dir containing data.yaml")
    ap.add_argument('--out', required=True, help="output ground_truth.json path")
    args = ap.parse_args()

    gt, warnings = convert(args.dataset)
    Path(args.out).write_text(json.dumps(gt, indent=2), encoding='utf-8')
    print(f"Wrote {len(gt)} entries to {args.out}")

    sizes = Counter((len(v['player1']), len(v['community']), len(v['player2']))
                    for v in gt.values())
    print("Row-size distribution (player1, community, player2):", dict(sizes))
    if warnings:
        print(f"{len(warnings)} warning(s) to review:")
        for w in warnings[:25]:
            print("  -", w)


if __name__ == '__main__':
    main()
