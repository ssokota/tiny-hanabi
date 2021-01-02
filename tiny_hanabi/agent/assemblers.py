"""Functions for assembling learners.
"""

from typing import Optional, Union

from .actors import ValueActor, PolicyActor
from ..game.settings import Settings
from .multi_agent_learners import (
    IndependentQLearner,
    IndependentReinforceLearner,
    IndependentA2CLearner,
    MultiAgentLearner,
    Sads,
)
from .parameters import ValueParameters, DoubleLrValueParameters, PolicyParameters
from .single_agent_learners import (
    QLearner,
    ReinforceLearner,
    A2CLearner,
    SingleAgentLearner,
    Algorithms,
)

learnertype = Union[SingleAgentLearner, MultiAgentLearner]


def make_ql(
    init_lr: float, init_epsilon: float, num_episodes: int, num_evals: int
) -> QLearner:
    """Make a Q-learner

    Args:
        init_lr: The initial learning rate (is linearly decayed)
        init_epsilon: The initial exploration rate (is linearly decayred)
        num_episodes: The number of episodes to train for
        num_evals: The number of evaluations to do

    Returns:
        Q-learner
    """
    return QLearner(
        ValueActor(ValueParameters(init_lr), init_epsilon), num_episodes, num_evals
    )


def make_hql(
    init_lr: float,
    init_lr2: float,
    init_epsilon: float,
    num_episodes: int,
    num_evals: int,
) -> QLearner:
    """Make a Hysteretic Q-Learner

    Hysteretic Q-learning is the same as Q-learning except that it uses
    a smaller learning rate if the target was underestimated.

    Args:
        init_lr: Initial overestimation learning rate (is linearly decayed)
        init_lr2: Initial underesimation learning rate (is linearly decayed)
        init_epsilon: The initial exploration rate (is linearly decayred)
        num_episodes: The number of episodes to train for
        num_evals: The number of evaluations to do

    Returns:
        HQ-learner
    """
    assert init_lr > init_lr2
    return QLearner(
        ValueActor(DoubleLrValueParameters(init_lr, init_lr2), init_epsilon),
        num_episodes,
        num_evals,
    )


def make_reinforce(
    init_lr: float, num_episodes: int, num_evals: int
) -> ReinforceLearner:
    """Make reinforce learner

    Args:
        init_lr: Initial learning rate (is linearly decayed)
        num_episodes: The number of episodes to train for
        num_evals: The number of evaluations to do

    Returns:
        Reinforce learner
    """
    return ReinforceLearner(
        PolicyActor(PolicyParameters(init_lr)), num_episodes, num_evals
    )


def make_a2c(
    pg_init_lr: float, ql_init_lr: float, num_episodes: int, num_evals: int
) -> A2CLearner:
    """Make advantage actor critic learner

    Args:
        pg_init_lr: Initial learning rate for actor (is linearly decayed)
        ql_init_lr: Initial learning rate for critic (is linearly decayed)
        num_episodes: The number of episodes to train for
        num_evals: The number of evaluations to do

    Returns:
        Advantage actor critic learner
    """
    return A2CLearner(
        PolicyActor(PolicyParameters(pg_init_lr)),
        ValueParameters(ql_init_lr),
        num_episodes,
        num_evals,
    )


def make_iql(
    init_lr: float,
    init_epsilon: float,
    sad: Optional[Sads],
    avd: bool,
    num_episodes: int,
    num_evals: int,
) -> IndependentQLearner:
    """Make independent Q-learners

    Args:
        init_lr: The initial learning rate (is linearly decayed)
        init_epsilon: The initial exploration rate (is linearly decayred)
        sad: Which sad variant to use (or none)
        avd: Whether to use additive value decomposition (aka VDN)
        num_episodes: The number of episodes to train for
        num_evals: The number of evaluations to do

    Returns:
        Independent Q-learner
    """
    alice = make_ql(init_lr, init_epsilon, num_episodes, num_evals)
    bob = make_ql(init_lr, init_epsilon, num_episodes, num_evals)
    return IndependentQLearner(alice, bob, sad, avd)


def make_ihql(
    init_lr: float,
    init_lr2: float,
    init_epsilon: float,
    sad: Optional[Sads],
    avd: bool,
    num_episodes: int,
    num_evals: int,
) -> IndependentQLearner:
    """Make independent hysteretic Q-learners

    Args:
        init_lr: Initial overestimation learning rate (is linearly decayed)
        init_lr2: Initial underesimation learning rate (is linearly decayed)
        init_epsilon: The initial exploration rate (is linearly decayred)
        sad: Which sad variant to use (or none)
        avd: Whether to use additive value decomposition (aka VDN)
        num_episodes: The number of episodes to train for
        num_evals: The number of evaluations to do

    Returns:
        Independent hysteretic Q-learner
    """
    alice = make_hql(init_lr, init_lr2, init_epsilon, num_episodes, num_evals)
    bob = make_hql(init_lr, init_lr2, init_epsilon, num_episodes, num_evals)
    return IndependentQLearner(alice, bob, sad, avd)


def make_ireinforce(
    init_lr: float, sad: Optional[Sads], num_episodes: int, num_evals: int
) -> IndependentReinforceLearner:
    """Make independent reinforce learners

    Args:
        init_lr: Initial learning rate (is linearly decayed)
        sad: Which sad variant to use (or none)
        num_episodes: The number of episodes to train for
        num_evals: The number of evaluations to do

    Returns:
        Independent reinforce learners
    """
    alice = make_reinforce(init_lr, num_episodes, num_evals)
    bob = make_reinforce(init_lr, num_episodes, num_evals)
    return IndependentReinforceLearner(alice, bob, sad)


def make_ia2c(
    pg_init_lr: float,
    ql_init_lr: float,
    sad: Optional[Sads],
    central_critic: bool,
    num_episodes: int,
    num_evals: int,
) -> IndependentA2CLearner:
    """Make independent advantage actor critic learners

    Args:
        pg_init_lr: Initial learning rate for actor (is linearly decayed)
        ql_init_lr: Initial learning rate for critic (is linearly decayed)
        sad: Which sad variant to use (or none)
        central_critic: Whether to use a central critic
        num_episodes: The number of episodes to train for
        num_evals: The number of evaluations to do

    Returns:
        Independet dvantage actor critic learners
    """
    alice = make_a2c(pg_init_lr, ql_init_lr, num_episodes, num_evals)
    bob = make_a2c(pg_init_lr, ql_init_lr, num_episodes, num_evals)
    if central_critic:
        alice.critic = bob.critic
    return IndependentA2CLearner(alice, bob, sad, central_critic)


def construct_learner(
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
) -> learnertype:
    """Makes learner from argument information

    Args:
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

    Returns:
        Learner instance
    """
    if isinstance(init_epsilon, float):
        assert 0 <= init_epsilon <= 1
    for lr in (pg_init_lr, ql_init_lr, ql_init_lr2):
        if isinstance(lr, float):
            assert lr > 0
    assert 1 <= num_evals <= num_episodes
    if setting.value != "decpomdp":
        assert not avd
        assert not central_critic
        assert not sad
    if algorithm.value == "ql":
        assert isinstance(ql_init_lr, float)
        assert isinstance(init_epsilon, float)
        assert ql_init_lr2 is None
        assert pg_init_lr is None
        assert not central_critic
        if setting.value == "decpomdp":
            return make_iql(ql_init_lr, init_epsilon, sad, avd, num_episodes, num_evals)
        return make_ql(ql_init_lr, init_epsilon, num_episodes, num_evals)
    if algorithm.value == "hql":
        assert isinstance(ql_init_lr, float)
        assert isinstance(ql_init_lr2, float)
        assert isinstance(init_epsilon, float)
        assert pg_init_lr is None
        assert not central_critic
        if setting.value == "decpomdp":
            return make_ihql(
                ql_init_lr, ql_init_lr2, init_epsilon, sad, avd, num_episodes, num_evals
            )
        return make_hql(ql_init_lr, ql_init_lr2, init_epsilon, num_episodes, num_evals)
    if algorithm.value == "reinforce":
        assert isinstance(pg_init_lr, float)
        assert ql_init_lr is None
        assert ql_init_lr2 is None
        assert init_epsilon is None
        assert not central_critic
        assert not avd
        if setting.value == "decpomdp":
            return make_ireinforce(pg_init_lr, sad, num_episodes, num_evals)
        return make_reinforce(pg_init_lr, num_episodes, num_evals)
    if algorithm.value == "a2c":
        assert isinstance(ql_init_lr, float)
        assert isinstance(pg_init_lr, float)
        assert ql_init_lr2 is None
        assert init_epsilon is None
        assert not avd
        if setting.value == "decpomdp":
            return make_ia2c(
                pg_init_lr, ql_init_lr, sad, central_critic, num_episodes, num_evals
            )
        return make_a2c(pg_init_lr, ql_init_lr, num_episodes, num_evals)
    raise (Exception)
