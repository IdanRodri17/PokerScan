"""
Integration tests for the /evaluate-winner endpoint — the code path the frontend
"Fix Detection" flow uses to re-evaluate a corrected board.

These drive the real FastAPI app through TestClient (routing, request validation,
the shared hand_evaluator, and the response model), and include a cross-path
consistency guard proving the /upload analyzer and /evaluate-winner agree on a
winner — the regression guard against the two evaluators diverging again.

Winners here are chosen from cases the evaluator scores correctly, so they are
unaffected by the three known scoring bugs documented (xfail) in
test_hand_evaluator.py.
"""

from fastapi.testclient import TestClient

from main import app
from ml.hand_evaluator import create_hand_evaluator
from ml.poker_game_analyzer import analyze_poker_game

client = TestClient(app)
ev = create_hand_evaluator()


def evaluate(player1, community, player2):
    """POST to /evaluate-winner and return (status_code, json)."""
    resp = client.post("/evaluate-winner", json={
        "player1_cards": player1,
        "community_cards": community,
        "player2_cards": player2,
    })
    return resp.status_code, resp.json()


# Shared, winner-agnostic board used by several cases.
BOARD = ['2h', '7d', '9c', 'Jd', '4s']


# ---------------------------------------------------------------------------
# Winner determination through the endpoint
# ---------------------------------------------------------------------------

def test_player1_wins():
    status, body = evaluate(['As', 'Ah'], BOARD, ['Ks', 'Qd'])  # pair of aces vs K-high
    assert status == 200
    ga = body['game_analysis']
    assert ga['tie'] is False
    assert ga['winner']['id'] == 1


def test_player2_wins():
    status, body = evaluate(['Ks', 'Qd'], BOARD, ['As', 'Ah'])  # mirror of above
    assert status == 200
    ga = body['game_analysis']
    assert ga['tie'] is False
    assert ga['winner']['id'] == 2


def test_tie_splits():
    # A 5-9 straight on the board; neither pair improves on it -> split pot.
    status, body = evaluate(['2c', '2d'], ['5h', '6d', '7c', '8s', '9h'], ['3c', '3d'])
    assert status == 200
    ga = body['game_analysis']
    assert ga['tie'] is True
    assert ga['winner'] is None
    assert len(ga['tied_players']) == 2


def test_kicker_decides_winner():
    # Both make a pair of aces; the winner is decided by the kicker (J beats T),
    # so the endpoint must compare real strength, not just hand category.
    status, body = evaluate(['As', 'Js'], ['Ah', 'Kd', 'Qc', '2s', '3h'], ['Ad', 'Td'])
    assert status == 200
    ga = body['game_analysis']
    assert ga['tie'] is False
    assert ga['winner']['id'] == 1
    # Both players genuinely hold a pair of aces.
    assert ga['players'][0]['hand_description'] == 'One Pair'
    assert ga['players'][1]['hand_description'] == 'One Pair'


def test_winner_depends_on_best_five():
    # P1 makes a king-high straight flush from the board; P2 only a pair of aces.
    status, body = evaluate(['Th', 'Jh'], ['9h', 'Kh', 'Qh', '2d', '7c'], ['As', 'Ad'])
    assert status == 200
    ga = body['game_analysis']
    assert ga['winner']['id'] == 1
    assert ga['winner']['winning_hand'] == 'Straight Flush'


# ---------------------------------------------------------------------------
# Response shape (must stay stable for the frontend)
# ---------------------------------------------------------------------------

def test_response_shape_preserved():
    status, body = evaluate(['As', 'Ah'], BOARD, ['Ks', 'Qd'])
    assert status == 200
    assert body['success'] is True
    ga = body['game_analysis']
    assert set(['community_cards', 'players', 'winner', 'tie', 'tied_players']).issubset(ga)
    assert ga['community_cards'] == BOARD
    assert len(ga['players']) == 2
    for p in ga['players']:
        assert set(['id', 'name', 'position', 'hole_cards', 'best_hand', 'hand_description']).issubset(p)
    assert ga['winner']['id'] in (1, 2)
    assert set(['id', 'name', 'winning_hand']).issubset(ga['winner'])


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def test_rejects_wrong_player_card_count():
    status, _ = evaluate(['As'], BOARD, ['Ks', 'Qd'])  # player 1 has only 1 card
    assert status == 400


def test_rejects_bad_community_count():
    status, _ = evaluate(['As', 'Ah'], ['2h', '7d'], ['Ks', 'Qd'])  # only 2 community cards
    assert status == 400


# ---------------------------------------------------------------------------
# Single-source-of-truth guards
# ---------------------------------------------------------------------------

def test_winner_matches_evaluator():
    # The endpoint's verdict must match hand_evaluator.compare_hands on the same cards.
    p1, community, p2 = ['As', 'Ah'], BOARD, ['Ks', 'Qd']
    _, body = evaluate(p1, community, p2)
    endpoint_winner = body['game_analysis']['winner']['id']
    verdict = ev.compare_hands(p1 + community, p2 + community)['winner']
    assert (endpoint_winner == 1 and verdict == 'hand1') or \
           (endpoint_winner == 2 and verdict == 'hand2')


def test_upload_analyzer_agrees_with_evaluator():
    """The /upload analyzer and the evaluator must pick the same winner.

    Drives analyze_poker_game (the /upload path: spatial grouping -> delegated
    evaluation -> winner) and cross-checks it against hand_evaluator directly.
    """
    height = 1000

    def det(card, x, y):
        return {'card_name': card, 'confidence': 0.9,
                'bbox': [x - 10, y - 10, x + 10, y + 10], 'center': [x, y]}

    p1 = ['As', 'Ks']
    community = ['Qh', 'Jd', 'Tc', '2s', '3h']
    p2 = ['2d', '7c']
    detections = (
        [det(c, 100 + 100 * i, 100) for i, c in enumerate(p1)] +        # top zone
        [det(c, 100 + 100 * i, 500) for i, c in enumerate(community)] +  # middle zone
        [det(c, 100 + 100 * i, 900) for i, c in enumerate(p2)]           # bottom zone
    )

    result = analyze_poker_game(detections, (height, 1000))

    # P1 holds A-K and the board has Q-J-T -> Broadway straight; P2 only a pair of 2s.
    assert result['winner'] is not None
    assert result['winner']['id'] == 1

    verdict = ev.compare_hands(p1 + community, p2 + community)['winner']
    assert verdict == 'hand1'  # evaluator agrees with the analyzer
