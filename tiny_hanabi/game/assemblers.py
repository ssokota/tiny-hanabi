"""Functions for assembling games.
"""

from typing import Union

import numpy as np

from .payoff_matrices import PAYOFFS, OPTIMAL_RETURNS, GameNames
from .settings import Settings, Game, DecPOMDP, PuBMDP, TBMDP, VBMDP


def get_game(gamename: GameNames, setting: Settings, normalize: bool = True) -> Game:
    """Return game `gamename`

    Args:
        gamename: Name of the game
        normalize: Whether to normalize the payoffs of the game to [0, 1]

    Returns:
        Instance of game `gamename`
    """
    payoffs = PAYOFFS[gamename.value]
    optimal_return = OPTIMAL_RETURNS[gamename.value]
    if normalize:
        maximum, minimum = payoffs.max(), payoffs.min()
        payoffs = normalize_payoffs(payoffs, maximum, minimum)
        optimal_return = normalize_payoffs(optimal_return, maximum, minimum)
    if setting.value == "decpomdp":
        return DecPOMDP(payoffs, optimal_return)
    if setting.value == "pubmdp":
        return PuBMDP(payoffs, optimal_return)
    if setting.value == "tbmdp":
        return TBMDP(payoffs, optimal_return)
    if setting.value == "vbmdp":
        return VBMDP(payoffs, optimal_return)
    raise (Exception)


def normalize_payoffs(
    data: Union[float, np.ndarray], maximum: float, minimum: float
) -> Union[float, np.ndarray]:
    """Normalize `data` to [0, 1] from [`minimum`, `maximum`]

    Args:
        data: Payoff matrix or optimal return value
        maximum: Maximum achievable score
        minimum: Minimum achievable score

    Returns:
        `data` normalized by [`minimum`, `maximum`] to [0, 1].
    """
    return (data - minimum) / (maximum - minimum)
