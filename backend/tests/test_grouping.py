"""
Tests for adaptive row-based card grouping in ml.poker_game_analyzer.

The analyzer clusters detected cards into up to 3 rows by the gaps between their
y-centers (no fixed 0.20/0.50/0.80 bands), identifies community vs player rows,
and reports a layout_confidence in [0, 1]. For ambiguous layouts the grouping may
be imperfect, so those cases assert EITHER correct grouping OR a low confidence.
"""

from ml.poker_game_analyzer import PokerGameAnalyzer, analyze_poker_game

H, W = 1000, 1000
LOW_CONFIDENCE = 0.6


def det(card_name, confidence, x, y):
    return {'card_name': card_name, 'confidence': confidence, 'center': [x, y]}


def group(detections):
    """Run grouping and return ({group: set(card_names)}, layout_confidence)."""
    groups, confidence = PokerGameAnalyzer()._group_cards_by_position(detections, (H, W))
    names = {k: {c['card_name'] for c in v} for k, v in groups.items()}
    return names, confidence


def test_clean_two_player_layout():
    names, conf = group([
        det('As', 0.95, 100, 150), det('Ks', 0.95, 200, 150),
        det('Qh', 0.9, 100, 500), det('Jd', 0.9, 200, 500), det('10c', 0.9, 300, 500),
        det('9s', 0.9, 400, 500), det('8h', 0.9, 500, 500),
        det('2d', 0.95, 100, 850), det('7c', 0.95, 200, 850),
    ])
    assert names['player1'] == {'As', 'Ks'}
    assert names['player2'] == {'2d', '7c'}
    assert names['community'] == {'Qh', 'Jd', '10c', '9s', '8h'}
    assert conf > 0.9


def test_tilted_layout_community_card_drifts_into_player_band():
    # Qh sits between player 1 (y=150) and the rest of the community (y=500),
    # so the rows do not separate cleanly.
    detections = [
        det('As', 0.95, 100, 150), det('Ks', 0.95, 200, 150),
        det('Qh', 0.85, 150, 300),  # drifted community card
        det('Jd', 0.9, 200, 500), det('10c', 0.9, 300, 500), det('9s', 0.9, 400, 500),
        det('2d', 0.95, 100, 850), det('7c', 0.95, 200, 850),
    ]
    names, conf = group(detections)
    community_correct = names['community'] == {'Qh', 'Jd', '10c', '9s'}
    # Either we recover the community exactly, or we flag the layout as ambiguous.
    assert community_correct or conf < LOW_CONFIDENCE


def test_preflop_no_community():
    names, conf = group([
        det('As', 0.9, 100, 200), det('Ks', 0.9, 200, 200),
        det('2d', 0.9, 100, 800), det('7c', 0.9, 200, 800),
    ])
    assert names['community'] == set()      # no community row invented
    assert names['player1'] == {'As', 'Ks'}
    assert names['player2'] == {'2d', '7c'}
    assert conf > 0.9


def test_noisy_layout_with_false_detection():
    # A clean 3-row layout plus a low-confidence false detection in the community band.
    names, conf = group([
        det('As', 0.95, 100, 150), det('Ks', 0.95, 200, 150),
        det('Qh', 0.9, 100, 500), det('Jd', 0.9, 200, 500), det('10c', 0.9, 300, 500),
        det('9s', 0.9, 400, 500), det('8h', 0.9, 500, 500),
        det('3h', 0.2, 250, 510),  # false detection, lowest confidence
        det('2d', 0.95, 100, 850), det('7c', 0.95, 200, 850),
    ])
    # The false card is dropped by the top-5 community filter -> correct grouping.
    assert '3h' not in names['community']
    assert names['community'] == {'Qh', 'Jd', '10c', '9s', '8h'}
    assert names['player1'] == {'As', 'Ks'}
    assert names['player2'] == {'2d', '7c'}


def test_community_only_single_row():
    names, conf = group([
        det('Qh', 0.9, 100, 500), det('Jd', 0.9, 200, 500), det('10c', 0.9, 300, 500),
        det('9s', 0.9, 400, 500), det('8h', 0.9, 500, 500),
    ])
    assert names['community'] == {'Qh', 'Jd', '10c', '9s', '8h'}
    assert names['player1'] == set()
    assert names['player2'] == set()
    assert conf == 1.0


def test_empty_detections():
    names, conf = group([])
    assert names['player1'] == set()
    assert names['community'] == set()
    assert names['player2'] == set()
    assert conf == 1.0


def test_grouping_is_order_independent():
    # With equal-confidence cards (incl. a 6th community card that must be cut to
    # 5), the grouping must be the same regardless of detection order.
    base = [
        det('As', 0.9, 100, 150), det('Ks', 0.9, 200, 150),
        det('Qh', 0.9, 100, 500), det('Jd', 0.9, 200, 500), det('3d', 0.9, 250, 500),
        det('10c', 0.9, 300, 500), det('9s', 0.9, 400, 500), det('8h', 0.9, 500, 500),
        det('2d', 0.9, 100, 850), det('7c', 0.9, 200, 850),
    ]
    ref_names, ref_conf = group(base)
    assert len(ref_names['community']) == 5  # one of the six was cut
    for perm in (list(reversed(base)), base[4:] + base[:4], base[7:] + base[:7]):
        names, conf = group(perm)
        assert names == ref_names
        assert conf == ref_conf


def test_layout_confidence_surfaced_in_result_dict():
    # The public analyze_poker_game result exposes layout_confidence for the frontend.
    detections = [
        det('As', 0.95, 100, 150), det('Ks', 0.95, 200, 150),
        det('Qh', 0.9, 100, 500), det('Jd', 0.9, 200, 500), det('10c', 0.9, 300, 500),
        det('9s', 0.9, 400, 500), det('8h', 0.9, 500, 500),
        det('2d', 0.95, 100, 850), det('7c', 0.95, 200, 850),
    ]
    result = analyze_poker_game(detections, (H, W))
    assert 'layout_confidence' in result
    assert 0.0 <= result['layout_confidence'] <= 1.0
    assert result['layout_confidence'] > 0.9
