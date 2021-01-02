"""Parameters. Parameter objects hold the parameters themselves, the learning
rates for the parameters, and methods for updating the parameters and
learning rates. Inheritance structure is:

BaseParameters
    -> Parameters
        -> ValueParameters : Holds parameters for state-action value function
        -> PolicyParameters : Holds logits for softmax policy
    -> DoubleLrParameters
        -> DoubleLrValueParameters : Holds parameters for state-action value
            function using two learning rates
"""
from abc import ABC, abstractmethod
from typing import Union, Dict

import numpy as np

from ..game.settings import statetype
from .utilities import softmax


class BaseParameters(ABC):
    """Base class for parameters

    Attributes:
        vals: Maps states to num_actions parameters
    """

    vals: Dict[statetype, np.ndarray]

    def add_state(self, state: statetype, num_legal_actions: int) -> None:
        """Adds `state` to `vals`

        Args:
            state: A player's information state
            num_legal_actions: Number of actions available to agent
        """
        if state not in self.vals:
            self.vals[state] = np.zeros(num_legal_actions)

    @abstractmethod
    def update_learning_rate(self, denominator: float) -> None:
        """Decay learning rate(s) as necessary

        Args:
            denominator: Timescale of decay
        """

    @abstractmethod
    def update_params(self, state: statetype, action: int, target: int) -> None:
        """Update parameters at `vals`[`state`] using `action` and `target`

        Args:
            state: A player's information state
            action: The action player took at `state`
            target: A target value
        """


class Parameters(BaseParameters):
    def __init__(self, init_lr: float) -> None:
        """Base class for learnable parameters with one learning rate

        Args:
            init_lr: Initial learning rate for parameter updates

        Attributes:
            See args, base class
            lr (float): Current learning rate for parameters updates
        """
        self.vals = {}
        self.init_lr = init_lr
        self.lr = init_lr

    @abstractmethod
    def update_params(self, state: statetype, action: int, target: float) -> None:
        """Update parameters at `vals`[`state`] using `action` and `target`

        Args:
            See base class
        """

    def update_learning_rate(self, denominator: float) -> None:
        """Decay `lr` by `init_lr` / `denominator`

        Args:
            See base class
        """
        self.lr -= self.init_lr / denominator


class DoubleLrParameters(BaseParameters):
    def __init__(self, init_lr: float, init_lr2: float) -> None:
        """Base class for learnable parameters with two learning rates

        Args:
            init_lr: Initial learning rate for parameter updates
            init_lr2: Initial learning rate for when target > estimate

        Attributes:
            See args, base class
            lr (float): Current learning rate for parameters updates
            lr2 (float): Current learning rate for when target > estimate
        """
        self.init_lr = init_lr
        self.init_lr2 = init_lr2
        self.lr = init_lr
        self.lr2 = init_lr2

    @abstractmethod
    def update_params(self, state: statetype, action: int, target: float) -> None:
        """Update parameters at `vals`[`state`] using `action` and `target`

        Args:
            See base class
        """

    def update_learning_rate(self, denominator: float) -> None:
        """Decay `lr` by `init_lr` / `denominator` (analogous for `lr2`)

        Args:
            See base class
        """
        self.lr -= self.init_lr / denominator
        self.lr2 -= self.init_lr2 / denominator


class ValueParameters(Parameters):
    """Value parameters with single learning rate

    Args:
        See base class

    Attributes:
        See base class
    """

    def update_params(self, state: statetype, action: int, target: float) -> None:
        """Move the `state`-`action` value toward `target`

        Args:
            See base class
        """
        self.vals[state][action] = gradient_update(
            self.vals[state][action], self.lr, target
        )


class DoubleLrValueParameters(DoubleLrParameters):
    """Value parameters with two learning rates

    Args:
        See base class

    Attribues:
        See base class
    """

    def update_params(self, state: statetype, action: int, target: float) -> None:
        """Move the `state`-`action` value toward `target` with appropriate lr

        Use `lr` when agent underestimated and `lr2` when agent overestimated.

        Args:
            See base class
        """
        est = self.vals[state][action]
        lr = self.lr if target >= est else self.lr2
        self.vals[state][action] = gradient_update(est, lr, target)


class PolicyParameters(Parameters):
    """Policy parameters

    Args:
        See base class

    Attribues:
        See base class
    """

    def update_params(self, state: statetype, action: int, target: float) -> None:
        """Do policy gradient update on logits

        grad_{logits_j} log pi(s, a_i) target =
            target * (1 - pi(s, a_j))    if j == i
            target * (0 - pi(s, a_j))    if j != i
        Note that `target` is NOT really a target.

        Args:
            See base class
        """
        logits = self.vals[state]
        probs = softmax(logits)
        for a_, p in enumerate(probs):
            grad_log_prob = int(a_ == action) - p
            self.vals[state][a_] = gradient_update(
                logits[a_], self.lr, grad_log_prob * target
            )


def gradient_update(value: float, lr: float, target: float) -> float:
    """Return average weighted by `lr` of `value` and `target`"""
    return (1 - lr) * value + lr * target


valueparametertype = Union[ValueParameters, DoubleLrValueParameters]
