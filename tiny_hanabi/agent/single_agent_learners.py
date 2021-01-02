"""Single-agent reinforcement learning algorithms. Supports Q-learning,
reinforce, and advantage actor critic. Inheritance structure is:

SingleAgentLearner
    -> QLearner
    -> ReinforceLearner
    -> A2CLearner

Q-learning update and actor critic updates are implemented as functions
for reuse across classes.
"""
from abc import ABC, abstractmethod
from enum import Enum
from typing import TypeVar, Generic, Union

from .actors import ValueActor, PolicyActor, Actor
from ..game.settings import statetype
from .parameters import valueparametertype
from .utilities import softmax


class Algorithms(Enum):
    q_learning = "ql"
    hysteretic_q_learning = "hql"
    reinforce = "reinforce"
    advantage_actor_critic = "a2c"

    def __str__(self):
        return self.value


T = TypeVar("T", ValueActor, PolicyActor)


class SingleAgentLearner(ABC, Generic[T]):
    def __init__(self, actor: T, num_episodes: int, num_evals: int) -> None:
        """Base class for single-agent learners

        Args:
            actor: The actor with which to select actions
            num_episodes: The number of episodes to train for
            num_evals: The number of evaluations to do
        Attributes:
            actor (T): The actor with which to select actions
            num_episodes (int): The number of episodes to train for
            num_evals (int): The number of evaluations to do
            eval_schedule (tuple): Episode indices on which to do evaluations
        """
        self.actor: T = actor
        self.num_episodes = num_episodes
        self.num_evals = num_evals
        self.eval_schedule = tuple(
            [
                (t * self.num_episodes) // (self.num_evals - 1)
                for t in range(self.num_evals)
            ]
        )

    def act(self, state: statetype, num_legal_actions: int, train: bool) -> int:
        """Pick an action given `state`

        If training, act normally. If evaluating, act greedily.

        Args:
            state: A player's information state
            num_legal_actions: The number of legal actions at `state`
            train: Whether we are training (or evaluating)

        Returns:
            Selected action
        """
        return self.actor.act(state, num_legal_actions, train)

    @abstractmethod
    def update_rates(self) -> None:
        """Update learning (and exploration) rates"""

    @abstractmethod
    def update_from_episode(self, episode: list) -> None:
        """Update actor from episode

        Args:
            episode: list of (state, action, reward) tuples
        """


class QLearner(SingleAgentLearner[ValueActor]):
    def __init__(self, actor: ValueActor, num_episodes: int, num_evals: int) -> None:
        """Q-learning agent

        Args:
            See base class
        Attributes:
            See base class
        """
        super().__init__(actor, num_episodes, num_evals)

    def update_from_episode(self, episode: list) -> None:
        """Update actor using Q-learning updates

        Args:
            See base class
        """
        q_update(self.actor.params, episode)

    def update_rates(self) -> None:
        """Update learning rate(s) and exploration rate"""
        self.actor.update_rates(self.num_episodes)


class ReinforceLearner(SingleAgentLearner[PolicyActor]):
    def __init__(self, actor: PolicyActor, num_episodes: int, num_evals: int) -> None:
        """Reinforce learning agent

        Args:
            See base class

        Attributes:
            See base class
        """
        super().__init__(actor, num_episodes, num_evals)

    def update_from_episode(self, episode: list) -> None:
        """Update actor with reinforce

        Args:
            See base class
        """
        states, actions, rewards = zip(*episode)
        payoff = sum(rewards)
        for s, a in zip(states, actions):
            self.actor.params.update_params(s, a, payoff)

    def update_rates(self) -> None:
        """Update learning rate of actor"""
        self.actor.update_rates(self.num_episodes)


class A2CLearner(SingleAgentLearner[PolicyActor]):
    def __init__(
        self,
        actor: PolicyActor,
        critic: valueparametertype,
        num_episodes: int,
        num_evals: int,
    ) -> None:
        """Advantage actor critic learner

        Args:
            See base class
            critic: Estimated state-action values

        Attributes:
            See base class
            critic (valueparametertype): Estimated state-action values
        """
        super().__init__(actor, num_episodes, num_evals)
        self.critic = critic

    def update_from_episode(self, episode: list) -> None:
        """Update actor using critic and critic using expected SARSA

        Args:
            See base class
        """
        states, actions, rewards = zip(*episode)
        for s, a, r, s_ in zip(states, actions, rewards, states[1:]):
            ac_actor_update(self.actor, self.critic, s, s, a)
            ac_critic_update(self.actor, self.critic, s, a, r, s_, s_)
        last_state = states[-1]
        last_action = actions[-1]
        last_reward = rewards[-1]
        ac_actor_update(self.actor, self.critic, last_state, last_state, last_action)
        self.critic.add_state(last_state, len(self.actor.params.vals[last_state]))
        self.critic.update_params(last_state, last_action, last_reward)

    def update_rates(self) -> None:
        """Update actor learning rate and critic learning rate"""
        self.actor.update_rates(self.num_episodes)
        # If we are using a centralized critic another learner may have already
        # updated the learning rate.
        if (self.critic.lr / self.critic.init_lr) > (
            self.actor.params.lr / self.actor.params.init_lr
        ):
            self.critic.update_learning_rate(self.num_episodes)


def q_update(params: valueparametertype, episode: list) -> None:
    """Perform a Q-value update on `params` given `episode`

    Q trained to match (s, a) to E[r + max_{a'} Q(s', a')]

    Args:
        params: The parameters to update
        episode: The episode from which to update the parameters
    """
    states, actions, rewards = zip(*episode)
    for s, a, r, s_ in zip(states, actions, rewards, states[1:]):
        params.update_params(s, a, r + params.vals[s_].max())
    params.update_params(states[-1], actions[-1], rewards[-1])


def ac_actor_update(
    actor: PolicyActor,
    critic: valueparametertype,
    a_state: statetype,
    c_state: statetype,
    action: int,
) -> None:
    """Perform actor critic update on actor

    pi trained to max E[log pi(s, a) A(s, a)]
    where A(s, a) = Q(s, a | pi) - V(s | pi)

    Args:
        actor: The actor
        critic: The critic
        a_state: The actor's information state
        c_state: The critic's information state. Same as `a_state` unless
            using a central critic.
        action: The action the actor took
    """
    critic.add_state(c_state, len(actor.params.vals[a_state]))
    policy = softmax(actor.params.vals[a_state])
    q_vals = critic.vals[c_state]
    actor.params.update_params(
        a_state, action, q_vals[action] - (q_vals * policy).sum()
    )


def ac_critic_update(
    actor: PolicyActor,
    critic: valueparametertype,
    c_state: statetype,
    action: int,
    reward: float,
    c_state_: statetype,
    a_state_: statetype,
) -> None:
    """Update critic using TD learning

    Q trained to match (s, a) to E[r + V(s' | pi)]
    using V(s' | pi) = pi(s') * Q(s')
    ie expected SARSA

    Args:
        actor: The actor
        critic: The critic
        c_state: The critic's information state
        action: The action the actor took
        c_state_: The critic's next information state
        a_state_: The actor's next information state. Same as `c_state_` unless
            using a central critic.
    """
    next_policy = softmax(actor.params.vals[a_state_])
    critic.add_state(c_state_, len(actor.params.vals[a_state_]))
    next_vals = critic.vals[c_state_]
    target = reward + (next_vals * next_policy).sum()
    critic.update_params(c_state, action, target)
