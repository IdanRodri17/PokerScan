"""
Photo-quality gate for uploaded poker photos.

PokerVision's main real-world failure is photos taken too far from the table — 9
small cards in a wide frame that the detector partly misses. This module turns the
signals the pipeline already produces into a simple "is this photo good enough?"
decision, so the UI can nudge the user to retake instead of showing a likely-wrong
result.

Signals (all already available after detection):
  * number of cards detected
  * median detected-box area relative to the image (a proxy for "cards too far/small")
  * the analyzer's layout_confidence (how cleanly the 3 rows separated)

Thresholds were calibrated against backend/eval/photos so that clearly-too-far or
poorly-laid-out photos flag while usable photos pass. The bar is deliberately
conservative: a false "retake" on a good photo is worse UX than letting a borderline
one through, so it must NOT fire on good photos.

No ML thresholds or detection logic live here.
"""

from statistics import median
from typing import Dict, List, Optional, Tuple

# --- Thresholds, calibrated against backend/eval/photos (see eval harness) ---
# Good photos detect >=8 cards with layout_confidence >=0.40; too-far photos drop to
# 4-6 cards, and messy layouts to <0.20. Card-box area barely varied across the eval
# set (all shot at similar distance), so its threshold is a conservative safety net
# for photos farther than anything in that set.
MIN_CARDS_DETECTED = 7            # a full hand shows ~9; far photos miss cards
MIN_MEDIAN_CARD_AREA_PCT = 0.045  # median card box as % of image area; tiny => too far
MIN_LAYOUT_CONFIDENCE = 0.20      # 0-1 row-separation score from the analyzer

_REASON_FEW_CARDS = (
    "Only {n} card{s} were detected (a full hand shows about 9). "
    "Move closer so the whole table fills the frame."
)
_REASON_SMALL_CARDS = (
    "The cards look small or far away — move closer so each card fills more of the frame."
)
_REASON_LOW_LAYOUT = (
    "The card rows aren't clearly separated — lay the cards out in 3 clear rows: "
    "your 2 cards on top, the 5 community cards in the middle, and the other player's "
    "2 cards on the bottom."
)


def _box_area(bbox) -> Optional[float]:
    """Area of a [x1, y1, x2, y2] box, or None if the box is missing/degenerate."""
    if not bbox or len(bbox) < 4:
        return None
    w = max(0.0, float(bbox[2]) - float(bbox[0]))
    h = max(0.0, float(bbox[3]) - float(bbox[1]))
    area = w * h
    return area if area > 0 else None


def assess_photo_quality(detections: List[Dict],
                         image_shape: Tuple[int, int],
                         layout_confidence: Optional[float]) -> Dict:
    """Assess whether an uploaded photo is good enough to trust.

    Args:
        detections: detection dicts, each optionally carrying a 'bbox' [x1,y1,x2,y2].
        image_shape: (height, width) of the image in pixels.
        layout_confidence: the analyzer's 0-1 row-separation score, or None.

    Returns:
        { ok: bool, reasons: [str], cards_detected: int,
          median_card_area_pct: float|None, layout_confidence: float|None }
    """
    height, width = image_shape
    image_area = max(1.0, float(height) * float(width))

    cards_detected = len(detections)

    areas = []
    for det in detections:
        bbox = det.get('bbox') if isinstance(det, dict) else None
        area = _box_area(bbox)
        if area is not None:
            areas.append(area)
    median_area_pct = (median(areas) / image_area * 100.0) if areas else None

    reasons: List[str] = []
    if cards_detected < MIN_CARDS_DETECTED:
        reasons.append(_REASON_FEW_CARDS.format(
            n=cards_detected, s="" if cards_detected == 1 else "s"))
    if median_area_pct is not None and median_area_pct < MIN_MEDIAN_CARD_AREA_PCT:
        reasons.append(_REASON_SMALL_CARDS)
    if layout_confidence is not None and layout_confidence < MIN_LAYOUT_CONFIDENCE:
        reasons.append(_REASON_LOW_LAYOUT)

    return {
        "ok": len(reasons) == 0,
        "reasons": reasons,
        "cards_detected": cards_detected,
        "median_card_area_pct": round(median_area_pct, 4) if median_area_pct is not None else None,
        "layout_confidence": round(float(layout_confidence), 3) if layout_confidence is not None else None,
    }
