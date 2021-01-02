"""Implements actors. Repository uses the term actor to mean an
action selector (given a state and some parameters). Inheritance
structure is:

Actor
    -> ValueActor : Makes decisions usings state-action value function
    -> PolicyActor : Makes decisions using policy
"""
from abc import ABC, abstractmethod

import numpy as np

from ..game.settings import statetype
from .parameters import valueparametertype, PolicyParameters, BaseParameters
from .utilities import softmax


class Actor(ABC):
    """Base class for actors

    Attributes:
        params: The actor's parameters
    """

    params: BaseParameters

    def act(self, state: statetype, num_legal_actions: int, train: bool) -> int:
        """Pick an action given `state`

        If training, act normally. If evaluating, act greedily. Note that for
        policy based methods, acting greedily is defined as taking the action
        with the largest logit.

        Args:
            state: A player's information state
            num_legal_actions: The number of legal actions at `state`
            train: Whether we are training (or evaluating)

        Returns:
            Selected action
        """
        self.params.add_state(state, num_legal_actions)
        if train:
            return self.act_normally(state)
        return self.act_greedily(state)

    @abstractmethod
    def act_normally(self, state: statetype) -> int:
        """Pick action balancing exploration/exploitation

        Args:
            state: Agent's information state

        Returns:
            Selected action
        """

    def act_greedily(self, state: statetype) -> int:
        """Pick most highly assessed action

        Args:
            state: Agent's information state

        Returns:
            Selected action
        """
        # Use deterministic argmax during evalution so that greedy policy is
        # consistent across rollouts. Otherwise, greedy policy may achieve
        # above optimal return or below most suboptimal return during
        # evaluation.
        return np.argmax(self.params.vals[state])

    @abstractmethod
    def update_rates(self, denominator: float) -> None:
        """Linearly decay learning (and exploration) rates

        Args:
            denominator: Timescale of decay
        """


class ValueActor(Actor):
    def __init__(self, params: valueparametertype, init_epsilon: float) -> None:
        """Actor using value parameters and epsilon-greedy exploration

        Args:
            params: State-action values
            init_epsilon: Initial exploration probability

        Attribues:
            params (valueparametertype): State-action values
            init_epsilon (float): Initial exploration probability
            epsilon (float): Current exploration probability
        """
        self.params: valueparametertype = params
        self.init_epsilon = init_epsilon
        self.epsilon = init_epsilon

    def act_normally(self, state: statetype) -> int:
        """Choose an action epsilon-greedily

        Args:
            state: Agent's information state

        Returns:
            Selected action
        """
        vals = self.params.vals[state]
        if np.random.random_sample() < self.epsilon:
            return np.random.choice(np.arange(len(vals)))
        # Important to use argmax with random tiebreaking during training
        return np.random.choice(np.flatnonzero(vals == vals.max()))

    def update_rates(self, denominator: float) -> None:
        """Linearly decay exploration and learning rate(s)

        Args:
            denominator: Timescale of decay
        """
        self.epsilon -= self.init_epsilon / denominator
        self.params.update_learning_rate(denominator)


class PolicyActor(Actor):
    def __init__(self, params: PolicyParameters) -> None:
        """Actor using policy parameters and softmax policy

        Args:
            params: Logits of softmax policy

        Attribues:
            params (PolicyParameters): Logits of softmax policy
        """
        self.params: PolicyParameters = params

    def act_normally(self, state: statetype) -> int:
        """Choose an action from softmax policy

        Args:
            state: Agent's information state

        Returns:
            Selected action
        """
        probs = softmax(self.params.vals[state])
        return np.random.choice(np.arange(len(probs)), p=probs)

    def update_rates(self, denominator: float) -> None:
        """Linearly decay exploration and learning rate

        Args:
            denominator: Timescale of decay
        """
        self.params.update_learning_rate(denominator)
