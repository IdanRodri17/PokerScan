"""
Unit tests for ml.hand_evaluator — the single source of truth for poker hand strength.

Both /upload (via PokerGameAnalyzer) and /evaluate-winner rank hands through this
module, so its correctness is what keeps the two paths in agreement.

NOTE ON xfail TESTS
-------------------
Writing these tests surfaced three real scoring bugs in the evaluator. By project
decision they are fixed in a separate follow-up, so the cases that assert the
*correct* poker behaviour are marked ``@pytest.mark.xfail(strict=True)``:

  1. Flush overflow      — a flush outranks full house / quads / straight flush /
                           royal flush (100**(4-i) multipliers in
                           _calculate_hand_strength).
  2. Wheel straight      — A-2-3-4-5 is scored as ace-high, so it ties Broadway
                           and beats lower straights (and the steel-wheel SF).
  3. One-pair kicker     — a high kicker can lift a lower pair above a higher pair.

``strict=True`` means these tests will START FAILING (XPASS) once the evaluator is
fixed — that is intentional: it's the signal to delete the xfail marker.
"""

import pytest

from ml.hand_evaluator import create_hand_evaluator, HandRank

ev = create_hand_evaluator()


def best(cards):
    return ev.evaluate_best_hand(cards)


def rank(cards):
    return best(cards).hand_rank


def strength(cards):
    return best(cards).hand_strength


# ---------------------------------------------------------------------------
# Classification: each 5-card hand is identified as the expected category
# ---------------------------------------------------------------------------

def test_high_card():
    # A K Q J 9, mixed suits, not a straight
    assert rank(['As', 'Kh', 'Qd', 'Jc', '9s']) == HandRank.HIGH_CARD


def test_one_pair():
    assert rank(['As', 'Ah', 'Qd', 'Jc', '9s']) == HandRank.ONE_PAIR


def test_two_pair():
    assert rank(['As', 'Ah', 'Qd', 'Qc', '9s']) == HandRank.TWO_PAIR


def test_three_of_a_kind():
    assert rank(['As', 'Ah', 'Ad', 'Qc', '9s']) == HandRank.THREE_OF_A_KIND


def test_straight_basic():
    assert rank(['6s', '7h', '8d', '9c', 'Ts']) == HandRank.STRAIGHT


def test_straight_wheel_is_classified_as_straight():
    # A-2-3-4-5: the ace plays low. Still a straight (only its *strength* is buggy).
    assert rank(['As', '2h', '3d', '4c', '5s']) == HandRank.STRAIGHT


def test_straight_broadway():
    assert rank(['Ts', 'Jh', 'Qd', 'Kc', 'As']) == HandRank.STRAIGHT


def test_flush():
    # A K Q J 9 all spades — not consecutive, so a flush rather than a straight flush
    assert rank(['As', 'Ks', 'Qs', 'Js', '9s']) == HandRank.FLUSH


def test_full_house():
    assert rank(['As', 'Ah', 'Ad', 'Kc', 'Ks']) == HandRank.FULL_HOUSE


def test_four_of_a_kind():
    assert rank(['As', 'Ah', 'Ad', 'Ac', 'Ks']) == HandRank.FOUR_OF_A_KIND


def test_straight_flush():
    assert rank(['5s', '6s', '7s', '8s', '9s']) == HandRank.STRAIGHT_FLUSH


def test_steel_wheel_is_classified_as_straight_flush():
    # A-2-3-4-5 suited is a straight flush, NOT a royal flush.
    assert rank(['As', '2s', '3s', '4s', '5s']) == HandRank.STRAIGHT_FLUSH


def test_royal_flush():
    assert rank(['Ts', 'Js', 'Qs', 'Ks', 'As']) == HandRank.ROYAL_FLUSH


# ---------------------------------------------------------------------------
# Category ordering (excluding the flush, which is covered separately below)
# ---------------------------------------------------------------------------

def test_category_strength_ordering():
    # Strengths must strictly increase as the hand category improves.
    sequence = [
        ['As', 'Kh', 'Qd', 'Jc', '9s'],   # high card
        ['2s', '2h', '3d', '4c', '6s'],   # one pair
        ['2s', '2h', '3d', '3c', '6s'],   # two pair
        ['2s', '2h', '2d', '3c', '6s'],   # three of a kind
        ['2s', '3h', '4d', '5c', '6s'],   # straight
        ['2s', '2h', '2d', '3c', '3s'],   # full house
        ['2s', '2h', '2d', '2c', '6s'],   # four of a kind
        ['5s', '6s', '7s', '8s', '9s'],   # straight flush
        ['Ts', 'Js', 'Qs', 'Ks', 'As'],   # royal flush
    ]
    strengths = [strength(c) for c in sequence]
    assert strengths == sorted(strengths)
    assert len(set(strengths)) == len(strengths)  # strictly increasing, no ties


def test_flush_beats_lower_ranked_hands():
    # The flush bug is an *upper-bound* overflow; it correctly beats weaker hands.
    flush = ['As', 'Ks', 'Qs', 'Js', '9s']
    assert strength(flush) > strength(['2s', '3h', '4d', '5c', '6s'])   # > straight
    assert strength(flush) > strength(['As', 'Ah', 'Ad', 'Qc', '9s'])   # > trips


# ---------------------------------------------------------------------------
# Tie-breakers / kickers (behaviour the evaluator gets right)
# ---------------------------------------------------------------------------

def test_high_card_kicker_decides():
    assert strength(['As', 'Kh', 'Qd', 'Jc', '9s']) > strength(['Ah', 'Ks', 'Qc', 'Jd', '8s'])


def test_one_pair_kicker_decides():
    # Same pair (KK), better kicker wins.
    assert strength(['Ks', 'Kh', 'Ad', 'Qc', 'Js']) > strength(['Kd', 'Kc', 'Qh', 'Js', '9s'])


def test_two_pair_higher_top_pair_wins():
    # Aces-up beats Kings-up even though the loser's second pair is higher.
    assert strength(['As', 'Ah', '3d', '3c', '9s']) > strength(['Ks', 'Kh', 'Qd', 'Qc', '2s'])


def test_two_pair_kicker_decides():
    # Same two pair (AA KK), kicker decides.
    assert strength(['As', 'Ah', 'Ks', 'Kh', 'Qd']) > strength(['Ac', 'Ad', 'Kc', 'Kd', 'Js'])


def test_trips_kicker_decides():
    # Same trips (AAA), kicker decides.
    assert strength(['As', 'Ah', 'Ad', 'Ks', 'Qc']) > strength(['Ac', 'Ah', 'Ad', 'Ks', 'Jc'])


def test_full_house_ranks_by_trips_then_pair():
    # Trips dominate: KKK22 beats QQQAA.
    assert strength(['Ks', 'Kh', 'Kd', '2c', '2s']) > strength(['Qs', 'Qh', 'Qd', 'Ac', 'As'])
    # Trips equal, the pair decides: KKKQQ beats KKK22.
    assert strength(['Ks', 'Kh', 'Kd', 'Qc', 'Qs']) > strength(['Kc', 'Kh', 'Kd', '2c', '2s'])


def test_flush_compares_high_cards():
    # Two flushes: the higher top card wins.
    assert strength(['As', 'Ks', 'Qs', 'Js', '9s']) > strength(['Ah', 'Kh', 'Qh', 'Jh', '8h'])


def test_four_of_a_kind_kicker_decides():
    # Same quads, kicker decides.
    assert strength(['As', 'Ah', 'Ad', 'Ac', 'Ks']) > strength(['As', 'Ah', 'Ad', 'Ac', 'Qs'])


def test_straight_flush_high_card_decides():
    # 9-high straight flush beats 6-high straight flush; royal beats king-high SF.
    assert strength(['5s', '6s', '7s', '8s', '9s']) > strength(['2s', '3s', '4s', '5s', '6s'])
    assert strength(['Ts', 'Js', 'Qs', 'Ks', 'As']) > strength(['9s', 'Ts', 'Js', 'Qs', 'Ks'])


# ---------------------------------------------------------------------------
# Exact ties
# ---------------------------------------------------------------------------

def test_identical_hands_tie():
    result = ev.compare_hands(['As', 'Ks', 'Qd', 'Jc', '9h'],
                              ['Ah', 'Kh', 'Qs', 'Jd', '9c'])
    assert result['valid'] is True
    assert result['winner'] == 'tie'


def test_board_plays_both_players_tie():
    # A royal flush on the board: neither player's hole cards matter -> tie.
    board = ['As', 'Ks', 'Qs', 'Js', 'Ts']
    result = ev.compare_hands(board + ['2c', '3d'], board + ['4c', '5d'])
    assert result['winner'] == 'tie'


# ---------------------------------------------------------------------------
# 7-card "pick the best 5" path (wrong winner if selection is buggy)
# ---------------------------------------------------------------------------

def test_seven_cards_picks_flush_over_pair():
    # A pair of aces is the "obvious" hand, but the real best 5 is a flush.
    assert rank(['As', 'Ah', '2s', '5s', '9s', 'Ks', '3d']) == HandRank.FLUSH


def test_seven_cards_finds_full_house():
    # Best 5 of 7 is kings full of aces; a naive pick would settle for trips/two pair.
    assert rank(['As', 'Ah', 'Kd', 'Kc', 'Ks', '3h', '3d']) == HandRank.FULL_HOUSE


def test_seven_cards_finds_straight_among_extras():
    assert rank(['2s', '3h', '4d', '5c', '6s', 'Kd', 'Ah']) == HandRank.STRAIGHT


def test_seven_cards_winner_decided_by_best_five():
    # Shared board; P1 makes a king-high straight flush, P2 only trip deuces.
    board = ['9h', 'Kh', 'Qh', '2d', '7c']
    p1 = board + ['Th', 'Jh']   # 9h-Th-Jh-Qh-Kh straight flush
    p2 = board + ['2s', '2h']   # trip deuces
    result = ev.compare_hands(p1, p2)
    assert result['winner'] == 'hand1'


def test_evaluate_returns_none_with_fewer_than_five_cards():
    assert ev.evaluate_best_hand(['As', 'Kh']) is None


# ---------------------------------------------------------------------------
# Public API surface
# ---------------------------------------------------------------------------

def test_evaluate_community_and_hole_cards_shape():
    result = ev.evaluate_community_and_hole_cards(['2h', '7d', '9c'], ['As', 'Ah'])
    assert result['valid'] is True
    assert result['hand_rank'] == 'One Pair'
    assert len(result['best_5_cards']) == 5
    assert 'hand_strength' in result


def test_compare_hands_returns_winner_and_tie():
    higher = ['As', 'Ah', 'Ad', 'Kc', 'Ks']   # full house
    lower = ['As', 'Ah', 'Qd', 'Jc', '9s']     # one pair
    assert ev.compare_hands(higher, lower)['winner'] == 'hand1'
    assert ev.compare_hands(lower, higher)['winner'] == 'hand2'
    assert ev.compare_hands(higher, list(higher))['winner'] == 'tie'


# ===========================================================================
# KNOWN-BUG cases — asserting CORRECT poker. xfail(strict) until the evaluator
# is fixed in the follow-up; they will XPASS (and fail the suite) once fixed.
# ===========================================================================

@pytest.mark.xfail(strict=True, reason="BUG: flush overflow — flush outranks full house")
def test_flush_loses_to_full_house():
    flush = ['As', 'Ks', 'Qs', 'Js', '9s']
    full_house = ['2s', '2h', '2d', '3c', '3s']
    assert strength(flush) < strength(full_house)


@pytest.mark.xfail(strict=True, reason="BUG: flush overflow — flush outranks royal flush")
def test_flush_loses_to_royal_flush():
    flush = ['As', 'Ks', 'Qs', 'Js', '9s']
    royal = ['Th', 'Jh', 'Qh', 'Kh', 'Ah']
    assert strength(flush) < strength(royal)


@pytest.mark.xfail(strict=True, reason="BUG: wheel scored as ace-high instead of 5-high")
def test_wheel_is_lowest_straight():
    wheel = ['As', '2h', '3d', '4c', '5s']
    six_high = ['2h', '3d', '4c', '5s', '6h']
    broadway = ['Ts', 'Jh', 'Qd', 'Kc', 'As']
    assert strength(six_high) > strength(wheel)
    assert strength(broadway) > strength(wheel)


@pytest.mark.xfail(strict=True, reason="BUG: steel-wheel straight flush scored as ace-high")
def test_steel_wheel_is_lowest_straight_flush():
    steel_wheel = ['As', '2s', '3s', '4s', '5s']
    six_high_sf = ['2h', '3h', '4h', '5h', '6h']
    assert strength(six_high_sf) > strength(steel_wheel)


@pytest.mark.xfail(strict=True, reason="BUG: one-pair kicker overflow — lower pair can outrank higher pair")
def test_higher_pair_beats_lower_pair_regardless_of_kicker():
    aces_low_kickers = ['As', 'Ah', '4d', '3c', '2s']
    kings_high_kickers = ['Ks', 'Kh', 'Ad', 'Qc', 'Js']
    assert strength(aces_low_kickers) > strength(kings_high_kickers)
