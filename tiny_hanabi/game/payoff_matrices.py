"""Payoff matrices for games. Payoff matrices must have shape:

(num_cards, num_cards, num_actions, num_actions)

They are indexed:

player 1's card, player 2's card, player 1's action, player 2's action

Also contains the optimal expected return for each game.

Games A-F are the tiny Hanabi suite.
"""

from enum import Enum

import numpy as np


class GameNames(Enum):
    A = "A"
    B = "B"
    C = "C"
    D = "D"
    E = "E"
    F = "F"
    G = "G"


PAYOFFS = {
    "A": np.array(
        [[[[0, 1], [0, 0]], [[0, 1], [3, 2]]], [[[3, 3], [3, 2]], [[2, 0], [3, 3]]]],
        dtype=np.float32,
    ),
    "B": np.array(
        [[[[1, 0], [1, 0]], [[0, 1], [0, 1]]], [[[0, 1], [0, 0]], [[1, 0], [1, 0]]]],
        dtype=np.float32,
    ),
    "C": np.array(
        [[[[3, 0], [0, 3]], [[2, 0], [3, 3]]], [[[2, 2], [3, 0]], [[0, 1], [0, 2]]]],
        dtype=np.float32,
    ),
    "D": np.array(
        [[[[3, 0], [1, 3]], [[3, 0], [3, 0]]], [[[3, 2], [0, 2]], [[0, 1], [0, 0]]]],
        dtype=np.float32,
    ),
    "E": np.array(
        [
            [[[10, 0, 0], [4, 8, 4], [10, 0, 0]], [[0, 0, 10], [4, 8, 4], [0, 0, 10]]],
            [[[0, 0, 10], [4, 8, 4], [0, 0, 0]], [[10, 0, 0], [4, 8, 4], [10, 0, 0]]],
        ],
        dtype=np.float32,
    ),
    "F": np.array(
        [
            [[[0, 3], [3, 2]], [[0, 0], [0, 1]], [[3, 1], [2, 1]]],
            [[[0, 2], [0, 1]], [[1, 2], [1, 2]], [[0, 1], [0, 3]]],
            [[[1, 3], [1, 2]], [[0, 3], [2, 2]], [[3, 1], [3, 0]]],
        ],
        dtype=np.float32,
    ),
    "G": np.array([[[[1, 2], [3, 4]]]], dtype=np.float32),
}

OPTIMAL_RETURNS = {
    "A": 2.25,
    "B": 1.00,
    "C": 2.50,
    "D": 2.50,
    "E": 10,
    "F": 2 + 1 / 3,
    "G": 4,
}
