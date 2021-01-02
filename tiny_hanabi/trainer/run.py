"""Functions for training.
"""

from typing import Optional

import numpy as np
from tqdm import tqdm

from ..agent.assemblers import learnertype
from ..game.settings import Game


def run(game: Game, learner: learnertype) -> list:
    """Train `learner` on `game`

    Returns:
        The expected returns of `learner` during training
    """
    expected_returns = [evaluate(game, learner)]
    for t in tqdm(range(1, learner.num_episodes + 1)):
        run_episode(game, learner)
        if t in learner.eval_schedule:
            expected_returns.append(evaluate(game, learner))
    return expected_returns


def evaluate(game: Game, learner: learnertype) -> float:
    """Evaluate `learner` performance

    Returns:
        The expected return of `learner` on `game`
    """
    returns = []
    for s0 in game.start_states():
        returns.append(run_episode(game, learner, train=False, s0=s0))
    return np.mean(returns)


def run_episode(game, learner, train: bool = True, s0: Optional[list] = None) -> float:
    """Run an episode of `learner` on `game`

    Args:
        train: True if learner is training, False for evaluation
        s0: The start state of the game, None means random start

    Returns:
        Epsiode payoff
    """
    game.reset(s0)
    while not game.is_terminal():
        a = learner.act(game.context(), game.num_legal_actions(), train)
        game.step(a)
    if train:
        learner.update_from_episode(game.episode())
        learner.update_rates()
    return game.payoff()
