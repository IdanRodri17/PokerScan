# PokerVision — real-photo accuracy report

_Generated: 2026-06-13 23:12:12_  
_Photos dir: `C:\Users\zionn\Desktop\PokerScan\backend\eval\photos` · Ground truth: `C:\Users\zionn\Desktop\PokerScan\backend\eval\ground_truth.json`_

**Photos evaluated: 3**

## Headline metrics

| Metric | Value | Detail |
|---|---|---|
| (a) Per-card detection (recall) | **0.0%** | 0/25 true cards read with the correct label |
| (a) Detection precision | n/a | 0/0 detected cards were real |
| (b) Grouping (given correct card) | **n/a** | 0/0 correctly-read cards put in the right group |
| (c) Correct winner named | **0.0%** | 0/3 photos with a determinable winner |
| (d) Human correction needed | **100.0%** | 3/3 photos not grouped perfectly |

## Most-confused classes

- Suit confusions (same rank, wrong suit): **0** (of which red H↔D: **0**)
- Rank confusions (same suit, wrong rank): **0**
- Pure misses (true card not detected at all): **25**
- False detections (card not in the photo): **0**

_No label confusions recorded._

## Per-photo results

| Photo | Cards read | Group ok | True→Named winner | Needs fix | Layout conf |
|---|---|---|---|---|---|
| sample_01.png | 0/9 | 0/0 | P1→NONE | yes | — |
| sample_02.png | 0/7 | 0/0 | P1→NONE | yes | — |
| sample_03.png | 0/9 | 0/0 | P1→NONE | yes | — |

## Notes & caveats

- **player1 = top hand, player2 = bottom hand** (the app's orientation). Grouping accuracy treats the two player hands as interchangeable; the winner metric does not (a swapped hand is a real end-user error).
- The confusion summary is **heuristic**: the ground truth has card labels per group but no per-card bounding boxes, so missed/spurious cards are paired by similarity to infer the most likely label slip.
- Detection (a) is label-only and ignores grouping; grouping (b) is conditioned on a card being read correctly, so the two are independent.
