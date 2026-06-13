#!/usr/bin/env python3
"""
Standalone accuracy harness for PokerVision on REAL phone photos.

The detector's headline ~98% mAP was measured on the clean Kaggle dataset it was
trained on -- not real phone photos in the app's actual layout. This harness
measures real-world accuracy by running the full production pipeline
(detect -> group -> analyze) via the existing ImageProcessor and comparing the
result to a hand-labelled ground truth.

It reports, SEPARATELY:
  (a) per-card detection accuracy  -- right 52-class label, ignoring grouping
  (b) grouping accuracy            -- given a correctly-read card, right group
  (c) end-to-end winner rate       -- named the correct winner
  (d) correction rate              -- how often a human "Fix Detection" was needed
plus a confusion summary of the most-confused classes (suit H<->D, rank slips).

Usage (from the backend/ directory, with the project venv):
    python eval/evaluate_real.py
    python eval/evaluate_real.py --photos-dir eval/photos --ground-truth eval/ground_truth.json

See eval/README.md for the ground-truth format and how to add labelled photos.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime
from io import BytesIO
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
BACKEND_DIR = EVAL_DIR.parent
sys.path.insert(0, str(BACKEND_DIR))

from services.image_processor import ImageProcessor      # noqa: E402
from ml.hand_evaluator import create_hand_evaluator      # noqa: E402

RANKS = {'A', 'K', 'Q', 'J', '10', '9', '8', '7', '6', '5', '4', '3', '2'}
SUITS = {'S', 'H', 'D', 'C'}
IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.heic', '.heif'}


# --------------------------------------------------------------------------- #
# Card normalisation
# --------------------------------------------------------------------------- #
def norm_card(card):
    """Canonical RANK+SUIT label: 'As'->'AS', 'Tc'/'10c'->'10C'. None if invalid."""
    if card is None:
        return None
    c = str(card).strip().upper()
    if not c:
        return None
    if c.startswith('10'):
        rank, suit = '10', c[2:3]
    elif c[0] == 'T':
        rank, suit = '10', c[1:2]
    else:
        rank, suit = c[0], c[1:2]
    if rank not in RANKS or suit not in SUITS:
        return None
    return rank + suit


def norm_list(cards):
    return [nc for nc in (norm_card(c) for c in (cards or [])) if nc]


def rank_of(card):
    return card[:-1]


def suit_of(card):
    return card[-1]


# --------------------------------------------------------------------------- #
# Pipeline output extraction
# --------------------------------------------------------------------------- #
def pipeline_groups(game_analysis):
    """{'player1': set, 'community': set, 'player2': set} from a game_analysis dict."""
    groups = {'player1': set(), 'community': set(), 'player2': set()}
    if not game_analysis:
        return groups
    groups['community'] = set(norm_list(game_analysis.get('community_cards', [])))
    for player in game_analysis.get('players', []):
        gid = 'player1' if player.get('id') == 1 else 'player2'
        groups[gid] = set(norm_list(player.get('hole_cards', [])))
    return groups


def pipeline_winner(game_analysis):
    if not game_analysis:
        return 'NONE'
    if game_analysis.get('tie'):
        return 'TIE'
    winner = game_analysis.get('winner')
    if winner:
        return 'P1' if winner.get('id') == 1 else 'P2'
    return 'NONE'


def true_winner(evaluator, gt):
    """Winner of the labelled hand, or None if not determinable (needs 2 + 3-5)."""
    p1, comm, p2 = gt['player1'], gt['community'], gt['player2']
    if len(p1) != 2 or len(p2) != 2 or not (3 <= len(comm) <= 5):
        return None
    result = evaluator.compare_hands(p1 + comm, p2 + comm)
    if not result.get('valid'):
        return None
    return {'hand1': 'P1', 'hand2': 'P2', 'tie': 'TIE'}[result['winner']]


# --------------------------------------------------------------------------- #
# Confusion pairing (heuristic: GT has labels-per-group but no per-card boxes)
# --------------------------------------------------------------------------- #
def _similarity(true_c, pred_c):
    if rank_of(true_c) == rank_of(pred_c):
        return 3   # suit confusion (same rank, wrong suit)
    if suit_of(true_c) == suit_of(pred_c):
        return 2   # rank confusion (same suit, wrong rank)
    return 1


def confusion_pairs(true_set, det_set):
    """Greedily pair undetected true cards with spurious detections to infer label
    slips. Returns (pairs[(true, pred)], pure_misses, false_positives)."""
    missed = sorted(true_set - det_set)
    extra = sorted(det_set - true_set)
    candidates = sorted(
        ((_similarity(t, p), t, p) for t in missed for p in extra), reverse=True
    )
    used_t, used_p, pairs = set(), set(), []
    for _score, t, p in candidates:
        if t in used_t or p in used_p:
            continue
        used_t.add(t)
        used_p.add(p)
        pairs.append((t, p))
    misses = [t for t in missed if t not in used_t]
    fps = [p for p in extra if p not in used_p]
    return pairs, misses, fps


# --------------------------------------------------------------------------- #
# Per-photo evaluation
# --------------------------------------------------------------------------- #
def _card_group(card, groups):
    for g in ('player1', 'community', 'player2'):
        if card in groups[g]:
            return g
    return 'unassigned'


def evaluate_photo(processor, evaluator, image_bytes, filename, gt):
    detection_results, _t, game_analysis, _v = processor.process_image(
        BytesIO(image_bytes), filename, analyze_game=True, create_visualization=False
    )

    det_all = set(norm_list(d.get('card') for d in (detection_results or [])))
    groups = pipeline_groups(game_analysis)

    gt_all = set(gt['player1'] + gt['community'] + gt['player2'])
    gt_groups = {
        'player1': set(gt['player1']),
        'community': set(gt['community']),
        'player2': set(gt['player2']),
    }

    # (a) detection -- label only, ignoring grouping
    correct = gt_all & det_all

    # (b) grouping given correct detection -- player1/player2 are interchangeable
    #     (they only mean top/bottom), so pick the player mapping that agrees more.
    straight = (len(groups['player1'] & gt_groups['player1'])
                + len(groups['player2'] & gt_groups['player2']))
    swapped = (len(groups['player1'] & gt_groups['player2'])
               + len(groups['player2'] & gt_groups['player1']))
    swap = swapped > straight
    grouped_correct = 0
    for c in correct:
        det_g = _card_group(c, groups)
        if swap:
            det_g = {'player1': 'player2', 'player2': 'player1'}.get(det_g, det_g)
        if det_g == _card_group(c, gt_groups):
            grouped_correct += 1

    # (c) winner
    tw = true_winner(evaluator, gt)
    pw = pipeline_winner(game_analysis)
    winner_evaluable = tw is not None
    winner_correct = winner_evaluable and pw == tw

    # (d) correction needed -- did the grouped output exactly match the truth?
    det_players = {frozenset(groups['player1']), frozenset(groups['player2'])}
    gt_players = {frozenset(gt_groups['player1']), frozenset(gt_groups['player2'])}
    perfect = groups['community'] == gt_groups['community'] and det_players == gt_players

    pairs, misses, fps = confusion_pairs(gt_all, det_all)

    return {
        'file': filename,
        'gt_cards': len(gt_all),
        'detected_correct': len(correct),
        'detected_total': len(det_all),
        'grouped_correct': grouped_correct,
        'grouped_total': len(correct),
        'winner_evaluable': winner_evaluable,
        'winner_correct': winner_correct,
        'true_winner': tw,
        'pipeline_winner': pw,
        'correction_needed': not perfect,
        'confusions': pairs,
        'misses': misses,
        'false_positives': fps,
        'layout_confidence': (game_analysis or {}).get('layout_confidence'),
    }


# --------------------------------------------------------------------------- #
# Aggregation & reporting
# --------------------------------------------------------------------------- #
def _pct(num, den):
    return f"{100 * num / den:.1f}%" if den else "n/a"


def summarize(records):
    conf = Counter()
    for r in records:
        for pair in r['confusions']:
            conf[pair] += 1
    suit_conf = {p: n for p, n in conf.items() if rank_of(p[0]) == rank_of(p[1])}
    rank_conf = {p: n for p, n in conf.items() if suit_of(p[0]) == suit_of(p[1])}
    red = {'H', 'D'}
    red_suit_conf = sum(n for (t, p), n in suit_conf.items()
                        if {suit_of(t), suit_of(p)} == red)
    return {
        'n_photos': len(records),
        'gt_cards': sum(r['gt_cards'] for r in records),
        'det_correct': sum(r['detected_correct'] for r in records),
        'det_total': sum(r['detected_total'] for r in records),
        'grouped_correct': sum(r['grouped_correct'] for r in records),
        'grouped_total': sum(r['grouped_total'] for r in records),
        'winner_evaluable': sum(1 for r in records if r['winner_evaluable']),
        'winner_correct': sum(1 for r in records if r['winner_correct']),
        'correction_needed': sum(1 for r in records if r['correction_needed']),
        'confusions': conf,
        'suit_conf_total': sum(suit_conf.values()),
        'rank_conf_total': sum(rank_conf.values()),
        'red_suit_conf': red_suit_conf,
        'misses': sum(len(r['misses']) for r in records),
        'false_positives': sum(len(r['false_positives']) for r in records),
    }


def render_report(s, records, args, missing_files, missing_gt):
    lines = []
    lines.append("# PokerVision — real-photo accuracy report")
    lines.append("")
    lines.append(f"_Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_  ")
    lines.append(f"_Photos dir: `{args.photos_dir}` · Ground truth: `{args.ground_truth}`_")
    lines.append("")
    lines.append(f"**Photos evaluated: {s['n_photos']}**")
    lines.append("")
    lines.append("## Headline metrics")
    lines.append("")
    lines.append("| Metric | Value | Detail |")
    lines.append("|---|---|---|")
    lines.append(f"| (a) Per-card detection (recall) | **{_pct(s['det_correct'], s['gt_cards'])}** "
                 f"| {s['det_correct']}/{s['gt_cards']} true cards read with the correct label |")
    lines.append(f"| (a) Detection precision | {_pct(s['det_correct'], s['det_total'])} "
                 f"| {s['det_correct']}/{s['det_total']} detected cards were real |")
    lines.append(f"| (b) Grouping (given correct card) | **{_pct(s['grouped_correct'], s['grouped_total'])}** "
                 f"| {s['grouped_correct']}/{s['grouped_total']} correctly-read cards put in the right group |")
    lines.append(f"| (c) Correct winner named | **{_pct(s['winner_correct'], s['winner_evaluable'])}** "
                 f"| {s['winner_correct']}/{s['winner_evaluable']} photos with a determinable winner |")
    lines.append(f"| (d) Human correction needed | **{_pct(s['correction_needed'], s['n_photos'])}** "
                 f"| {s['correction_needed']}/{s['n_photos']} photos not grouped perfectly |")
    lines.append("")
    lines.append("## Most-confused classes")
    lines.append("")
    lines.append(f"- Suit confusions (same rank, wrong suit): **{s['suit_conf_total']}** "
                 f"(of which red H↔D: **{s['red_suit_conf']}**)")
    lines.append(f"- Rank confusions (same suit, wrong rank): **{s['rank_conf_total']}**")
    lines.append(f"- Pure misses (true card not detected at all): **{s['misses']}**")
    lines.append(f"- False detections (card not in the photo): **{s['false_positives']}**")
    lines.append("")
    if s['confusions']:
        lines.append("| True card | Detected as | Count |")
        lines.append("|---|---|---|")
        for (t, p), n in s['confusions'].most_common(15):
            lines.append(f"| {t} | {p} | {n} |")
    else:
        lines.append("_No label confusions recorded._")
    lines.append("")
    lines.append("## Per-photo results")
    lines.append("")
    lines.append("| Photo | Cards read | Group ok | True→Named winner | Needs fix | Layout conf |")
    lines.append("|---|---|---|---|---|---|")
    for r in records:
        conf = r['layout_confidence']
        conf_s = f"{conf:.2f}" if isinstance(conf, (int, float)) else "—"
        lines.append(
            f"| {r['file']} | {r['detected_correct']}/{r['gt_cards']} "
            f"| {r['grouped_correct']}/{r['grouped_total']} "
            f"| {r['true_winner'] or '—'}→{r['pipeline_winner']} "
            f"| {'yes' if r['correction_needed'] else 'no'} | {conf_s} |"
        )
    lines.append("")
    if missing_files or missing_gt:
        lines.append("## Data warnings")
        lines.append("")
        for f in missing_files:
            lines.append(f"- Ground truth lists `{f}` but the image file is missing.")
        for f in missing_gt:
            lines.append(f"- Image `{f}` has no ground-truth entry (skipped).")
        lines.append("")
    lines.append("## Notes & caveats")
    lines.append("")
    lines.append("- **player1 = top hand, player2 = bottom hand** (the app's orientation). "
                 "Grouping accuracy treats the two player hands as interchangeable; the winner "
                 "metric does not (a swapped hand is a real end-user error).")
    lines.append("- The confusion summary is **heuristic**: the ground truth has card labels per "
                 "group but no per-card bounding boxes, so missed/spurious cards are paired by "
                 "similarity to infer the most likely label slip.")
    lines.append("- Detection (a) is label-only and ignores grouping; grouping (b) is conditioned "
                 "on a card being read correctly, so the two are independent.")
    return "\n".join(lines) + "\n"


def console_summary(s):
    return (
        "================ PokerVision real-photo accuracy ================\n"
        f"  Photos evaluated         : {s['n_photos']}\n"
        f"  (a) Detection (recall)   : {_pct(s['det_correct'], s['gt_cards'])}"
        f"   ({s['det_correct']}/{s['gt_cards']} cards)\n"
        f"      precision            : {_pct(s['det_correct'], s['det_total'])}\n"
        f"  (b) Grouping (correct)   : {_pct(s['grouped_correct'], s['grouped_total'])}"
        f"   ({s['grouped_correct']}/{s['grouped_total']})\n"
        f"  (c) Correct winner       : {_pct(s['winner_correct'], s['winner_evaluable'])}"
        f"   ({s['winner_correct']}/{s['winner_evaluable']})\n"
        f"  (d) Correction needed    : {_pct(s['correction_needed'], s['n_photos'])}\n"
        f"  Confusions: suit {s['suit_conf_total']} (red H/D {s['red_suit_conf']}),"
        f" rank {s['rank_conf_total']}, misses {s['misses']}, false {s['false_positives']}\n"
        "================================================================"
    )


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def load_ground_truth(path):
    raw = json.loads(Path(path).read_text(encoding='utf-8'))
    return {
        fname: {
            'player1': norm_list(entry.get('player1')),
            'community': norm_list(entry.get('community')),
            'player2': norm_list(entry.get('player2')),
        }
        for fname, entry in raw.items()
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate PokerVision on real photos.")
    parser.add_argument('--photos-dir', default=str(EVAL_DIR / 'photos'))
    parser.add_argument('--ground-truth', default=str(EVAL_DIR / 'ground_truth.json'))
    parser.add_argument('--report', default=str(EVAL_DIR / 'report.md'))
    args = parser.parse_args()

    ground_truth = load_ground_truth(args.ground_truth)
    photos_dir = Path(args.photos_dir)
    files = ({p.name: p for p in photos_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS}
             if photos_dir.exists() else {})

    print(f"Loading ImageProcessor (model)…")
    processor = ImageProcessor()
    if not processor.ml_enabled:
        print("⚠️  ML model NOT loaded — detection falls back to mock data; "
              "results would be meaningless. Aborting.")
        sys.exit(1)
    evaluator = create_hand_evaluator()

    print(f"Evaluating {len(ground_truth)} labelled photo(s) from {photos_dir}…")
    records, missing_files = [], []
    for fname in sorted(ground_truth):
        if fname not in files:
            missing_files.append(fname)
            continue
        try:
            data = files[fname].read_bytes()
            rec = evaluate_photo(processor, evaluator, data, fname, ground_truth[fname])
            records.append(rec)
            print(f"  ✓ {fname}: read {rec['detected_correct']}/{rec['gt_cards']} cards, "
                  f"winner true={rec['true_winner']} named={rec['pipeline_winner']}")
        except Exception as e:  # keep going on a bad photo
            print(f"  ✗ {fname}: pipeline error: {e}")
    missing_gt = [f for f in files if f not in ground_truth]

    summary = summarize(records)
    report = render_report(summary, records, args, missing_files, missing_gt)
    Path(args.report).write_text(report, encoding='utf-8')

    print("\n" + console_summary(summary))
    print(f"\nReport written to {args.report}")
    if missing_files:
        print(f"⚠️  {len(missing_files)} labelled photo(s) had no image file.")
    if missing_gt:
        print(f"ℹ️  {len(missing_gt)} image(s) had no ground-truth entry (skipped).")


if __name__ == '__main__':
    main()
