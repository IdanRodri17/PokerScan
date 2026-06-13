"""
Tests for the photo-quality retake gate (services/photo_quality.py).

Synthetic detections: a clean 9-card layout passes; too-far / too-few / too-small /
messy-layout photos flag — and the gate must never crash on missing signals.
"""

from services.photo_quality import (
    assess_photo_quality,
    MIN_CARDS_DETECTED,
    MIN_MEDIAN_CARD_AREA_PCT,
    MIN_LAYOUT_CONFIDENCE,
)

H, W = 1000, 1000  # image area = 1,000,000 px


def card(i, w=40, h=60):
    """A detection with a reasonably large box (40x60 = 0.24% of the image)."""
    x, y = 50 + (i % 3) * 120, 100 + (i // 3) * 250
    return {"card": "As", "confidence": 0.9, "bbox": [x, y, x + w, y + h]}


def nine_clean():
    return [card(i) for i in range(9)]


def test_clean_layout_passes():
    q = assess_photo_quality(nine_clean(), (H, W), 0.6)
    assert q["ok"] is True
    assert q["reasons"] == []
    assert q["cards_detected"] == 9
    assert q["median_card_area_pct"] > MIN_MEDIAN_CARD_AREA_PCT


def test_too_few_cards_flags():
    q = assess_photo_quality(nine_clean()[:4], (H, W), 0.6)  # only 4 cards
    assert q["ok"] is False
    assert q["cards_detected"] == 4
    assert any("detected" in r for r in q["reasons"])


def test_small_far_cards_flag():
    # 9 cards but each box is 6x6 = 36 px (0.0036%) << the area threshold.
    tiny = [card(i, w=6, h=6) for i in range(9)]
    q = assess_photo_quality(tiny, (H, W), 0.6)
    assert q["ok"] is False
    assert any("far" in r or "small" in r for r in q["reasons"])


def test_low_layout_confidence_flags():
    q = assess_photo_quality(nine_clean(), (H, W), 0.05)  # rows not separated
    assert q["ok"] is False
    assert any("rows" in r for r in q["reasons"])


def test_multiple_reasons_combine():
    # Few + tiny + messy: all three reasons fire.
    bad = [card(i, w=5, h=5) for i in range(4)]
    q = assess_photo_quality(bad, (H, W), 0.05)
    assert q["ok"] is False
    assert len(q["reasons"]) == 3


def test_passes_with_missing_bboxes():
    # No bbox -> area check is skipped; 9 cards + good layout still passes.
    dets = [{"card": "As", "confidence": 0.9} for _ in range(9)]
    q = assess_photo_quality(dets, (H, W), 0.6)
    assert q["ok"] is True
    assert q["median_card_area_pct"] is None


def test_none_layout_confidence_skips_layout_check():
    q = assess_photo_quality(nine_clean(), (H, W), None)
    assert q["ok"] is True
    assert q["layout_confidence"] is None


def test_thresholds_are_sane():
    assert MIN_CARDS_DETECTED == 7
    assert 0 < MIN_MEDIAN_CARD_AREA_PCT < 1
    assert 0 < MIN_LAYOUT_CONFIDENCE < 1
