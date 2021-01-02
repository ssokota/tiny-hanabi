"""Interface for running experiments.
"""

import argparse
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from tiny_hanabi.agent.assemblers import construct_learner, learnertype
from tiny_hanabi.agent.single_agent_learners import Algorithms
from tiny_hanabi.agent.multi_agent_learners import Sads
from tiny_hanabi.game.settings import Game, Settings
from tiny_hanabi.game.payoff_matrices import GameNames
from tiny_hanabi.game.assemblers import get_game
from tiny_hanabi.trainer.run import run

Path("results").mkdir(exist_ok=True)
Path("results/data").mkdir(exist_ok=True)
Path("results/figures").mkdir(exist_ok=True)


def save_and_plot(
    expected_returns: list,
    game: Game,
    learner: learnertype,
    gamename: GameNames,
    setting: Settings,
    algorithm: Algorithms,
    pg_init_lr: Optional[float],
    ql_init_lr: Optional[float],
    ql_init_lr2: Optional[float],
    init_epsilon: Optional[float],
    avd: bool,
    central_critic: bool,
    sad: Optional[Sads],
    num_episodes: int,
    num_evals: int,
    fn: str,
    plot: bool,
) -> None:
    """Save (and plot) data

    Args:
        expected_returns: List of expected returns
        game: Game object
        gamename: Name of Tiny Hanabi game
        setting: Name of setting
        algorithm: Name of algorithm
        ql_init_lr: Initial Q-learning learning rate, is linearly decayed
        ql_init_lr2: Second Q-learning learning for HQL, used when
            target < estimate, is linearly decayed
        init_epsilon: Epsilon-greedy exploration rate, is linearly decayed
        avd: Whether to used additive value decomposition (aka VDN)
        central_critic: Whether to use a centralized value function
        sad: Whether to use a sad variant
        num_episodes: How many episodes to train for
        num_evals: How many evaluations to do
        fn: filename to which to save results
        plot: whether to plot results
    """
    df = pd.DataFrame(
        {
            "episode": learner.eval_schedule,
            "expected_return": expected_returns,
            "optimal_return": num_evals * [game.optimal_return],
            "gamename": num_evals * [gamename.value],
            "setting": num_evals * [setting.value],
            "algorithm": num_evals * [algorithm.value],
            "pg_init_lr": num_evals * [pg_init_lr],
            "ql_init_lr": num_evals * [ql_init_lr],
            "ql_init_lr2": num_evals * [ql_init_lr2],
            "init_epsilon": num_evals * [init_epsilon],
            "avd": num_evals * [avd],
            "central_critic": num_evals * [central_critic],
            "sad": num_evals * [sad.value if sad else None],
        }
    )
    df.to_pickle("results/data/" + fn + ".pkl")
    if plot:
        plt.axhline(y=game.optimal_return, color="gray", linestyle="-", linewidth=1)
        sns.lineplot(data=df, x="episode", y="expected_return")
        algorithm = "a2c2" if central_critic else algorithm.value
        title = f"Game {gamename.value}; {setting.value}; {algorithm};"
        if avd:
            title += " avd;"
        if sad:
            title += f" {sad.value};"
        if pg_init_lr:
            title += f" pglr={pg_init_lr};"
        if ql_init_lr:
            title += f" qllr={ql_init_lr};"
        if ql_init_lr2:
            title += f" qllr2={ql_init_lr2};"
        if init_epsilon:
            title += f" eps={init_epsilon};"
        plt.title(title[:-1])
        plt.savefig("results/figures/" + fn + ".pdf")


def interface(
    gamename: GameNames,
    setting: Settings,
    algorithm: Algorithms,
    pg_init_lr: Optional[float] = None,
    ql_init_lr: Optional[float] = None,
    ql_init_lr2: Optional[float] = None,
    init_epsilon: Optional[float] = None,
    avd: bool = False,
    central_critic: bool = False,
    sad: Optional[Sads] = None,
    num_episodes: int = 1_000_000,
    num_evals: int = 100,
    fn: str = "example",
    plot: bool = False,
) -> None:
    """User interface for running experiments

    Args:
        gamename: Name of Tiny Hanabi game
        setting: Name of setting
        algorithm: Name of algorithm
        ql_init_lr: Initial Q-learning learning rate, is linearly decayed
        ql_init_lr2: Second Q-learning learning for HQL, used when
            target < estimate, is linearly decayed
        init_epsilon: Epsilon-greedy exploration rate, is linearly decayed
        avd: Whether to used additive value decomposition (aka VDN)
        central_critic: Whether to use a centralized value function
        sad: Whether to use a sad variant
        num_episodes: How many episodes to train for
        num_evals: How many evaluations to do
        fn: filename to which to save results
        plot: whether to plot results
    """
    game = get_game(gamename, setting)
    learner = construct_learner(
        setting,
        algorithm,
        pg_init_lr,
        ql_init_lr,
        ql_init_lr2,
        init_epsilon,
        avd,
        central_critic,
        sad,
        num_episodes,
        num_evals,
    )
    expected_returns = run(game, learner)
    save_and_plot(
        expected_returns,
        game,
        learner,
        gamename,
        setting,
        algorithm,
        pg_init_lr,
        ql_init_lr,
        ql_init_lr2,
        init_epsilon,
        avd,
        central_critic,
        sad,
        num_episodes,
        num_evals,
        fn,
        plot,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("gamename", type=GameNames, choices=list(GameNames))
    parser.add_argument("setting", type=Settings, choices=list(Settings))
    parser.add_argument("algorithm", type=Algorithms, choices=list(Algorithms))
    parser.add_argument("--pg_init_lr", type=float)
    parser.add_argument("--ql_init_lr", type=float)
    parser.add_argument("--ql_init_lr2", type=float)
    parser.add_argument("--init_epsilon", type=float)
    parser.add_argument("--avd", default=False, action="store_true")
    parser.add_argument("--central_critic", default=False, action="store_true")
    parser.add_argument("--sad", default=None, type=Sads, choices=list(Sads))
    parser.add_argument("--num_episodes", type=int, default=int(1e6))
    parser.add_argument("--num_evals", type=int, default=100)
    parser.add_argument("--fn", default="example")
    parser.add_argument("--plot", default=False, action="store_true")
    args = parser.parse_args()
    interface(**vars(args))
