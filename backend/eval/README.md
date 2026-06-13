# Real-photo accuracy evaluation

The detector's headline **~98% mAP** was measured on the clean Kaggle dataset it
was trained on — not on real phone photos in the app's actual layout. Before
retraining anything, use this harness to measure how the **full pipeline**
(detect → group → analyze) actually performs on real photos.

`evaluate_real.py` runs each labelled photo through the production
`ImageProcessor` and reports, **separately**:

| Metric | What it answers |
|---|---|
| **(a) detection** | Did the detector read the right 52-class label? (recall + precision, *ignoring* grouping) |
| **(b) grouping** | Given a correctly-read card, was it put in the right group? |
| **(c) winner** | Did the app name the correct winner end-to-end? |
| **(d) correction** | How often would a human have needed to hit "Fix Detection"? |

…plus a **confusion summary** of the most-confused classes (suit H↔D, rank slips).

## Run it

From the `backend/` directory, using the project venv:

```bash
python eval/evaluate_real.py
# or with explicit paths:
python eval/evaluate_real.py --photos-dir eval/photos --ground-truth eval/ground_truth.json --report eval/report.md
```

It prints a summary and writes [`report.md`](report.md). It **aborts** if the ML
model isn't loaded (so you never measure mock detections by accident).

> The repo ships with 3 **blank placeholder photos** so the harness runs
> end-to-end with no real data. They contain no cards, so every metric reads 0% /
> "correction needed" until you replace them with real photos — that's expected.

## Add your own labelled photos

1. Drop the photos into `eval/photos/` (`.jpg`, `.jpeg`, `.png`, `.heic`, … are all read).
2. Add one entry per photo to `eval/ground_truth.json`, keyed by **filename**:

```json
{
  "table_001.jpg": {
    "player1": ["As", "Ks"],
    "community": ["Qh", "Jd", "10c", "9s", "8h"],
    "player2": ["2d", "7c"]
  }
}
```

### Conventions

- **`player1` = the TOP hand, `player2` = the BOTTOM hand** — matches the app's
  orientation. Get this right: grouping accuracy treats the two player hands as
  interchangeable, but the **winner** metric does not (a swapped hand is a real
  end-user error worth catching).
- **`community`** = 3 (flop), 4 (turn), or 5 (river) cards. Preflop (no community)
  is allowed — leave it `[]`.
- **Card notation:** rank + suit. Suit ∈ `s h d c` (case-insensitive). Ten may be
  `10` or `T`. So `As`, `AS`, `10c`, `Tc` are all accepted.

## Converting a Roboflow / YOLO export

If your photos already have YOLO bounding-box labels (a Roboflow export with
`images/train/`, `labels/train/`, `data.yaml`), you don't need to re-label by
hand — `labels_to_ground_truth.py` derives the grouped ground truth from the box
positions:

```bash
python eval/labels_to_ground_truth.py \
    --dataset eval/photos/<your_export>/<your_export> \
    --out eval/ground_truth.json
```

It clusters each photo's boxes into top/middle/bottom rows (player1 / community /
player2) and prints the row-size distribution plus any photo that didn't form a
clean 2 / 3–5 / 2 layout.

> ⚠️ **Verify the labels actually match the images.** A YOLO `.txt` only matches
> its `.jpg` if they came from the same annotation run. If you combine images from
> one place with labels from another, the filenames can line up while the contents
> don't — the harness then reports a low "detection" score that is really a
> ground-truth mismatch, **not** a model failure. Open a few photos and check them
> against the generated `ground_truth.json` before trusting any number.

## How many photos?

- **Start** with ~30–50 labelled photos for a first directional read.
- **Baseline:** use all **100–150** — that gives a per-card detection estimate to
  ≈ ±2–3% and a winner-rate estimate to ≈ ±7–8%.
- **Comparing models** (e.g. "did retraining help by 5%?"): aim for **250–300+**.

**Diversity beats volume.** Spread photos across lighting (bright/dim/glare),
tilt/angle, phone models, backgrounds/felt, card decks, distance, and partial
occlusion. Try to make each of the 52 cards appear ≥ ~10 times so the confusion
summary is meaningful, and **deliberately include hard cases** (tilt, pocket
pairs, glare) — that's where the real failures hide.

## Caveats

- The confusion summary is **heuristic**: the ground truth has labels per group
  but no per-card bounding boxes, so missed/spurious cards are paired by
  similarity to infer the most likely label slip.
- Detection (a) is label-only; grouping (b) is conditioned on a correct read — so
  the two numbers are independent and you can tell *which* stage is the bottleneck.
