"""Multi-agent learning algorithms. Supports each single-agent learning
algorithm playing independently with itself and also supports simplified
action decoding, additive value decomposition (aka VDN), and centralized
value functions. Inheritance structure is:

MultiAgentLearner
    -> IndependentQLearner
    -> IndependentReinforceLearner
    -> IndependentA2CLearner

Independent updates, additive value decomposition updates, and simplified
action decoding updates are implemented as functions for reuse across classes.
Centralized value function updates are implemented within the
IndependentA2CLearner, as it is the only class that uses them.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional, TypeVar, Generic

from .actors import PolicyActor
from .single_agent_learners import (
    SingleAgentLearner,
    QLearner,
    ReinforceLearner,
    A2CLearner,
    ac_actor_update,
    ac_critic_update,
)

T = TypeVar("T", QLearner, ReinforceLearner, A2CLearner)


class Sads(Enum):
    sad = "sad"
    bsad = "bsad"
    asad = "asad"
    psad = "psad"


class MultiAgentLearner(ABC, Generic[T]):
    def __init__(self, alice: T, bob: T, sad: Optional[Sads]):
        """Base class for independent learning

        Args:
            alice: Player 1
            bob: Player 2
            sad: Which simplified action decoding variant to use (or none)

        Attributes:
            alice (T): Player 1
            bob (T): Player 2
            sad (Optional[Sads]): Which simplified action decoding variant
                to use (or none)
            num_episodes (int): The number of episodes to train for
            num_evals (int): The number of evaluations to do
            eval_schedule (tuple): Episode indices on which to do evaluations
        """
        self.alice: T = alice
        self.bob: T = bob
        self.sad = sad
        assert alice.num_episodes == bob.num_episodes
        self.num_episodes = alice.num_episodes
        assert alice.num_evals == bob.num_evals
        self.num_evals = alice.num_evals
        self.eval_schedule = alice.eval_schedule

    def act(self, context: tuple, num_legal_actions: int, train: bool) -> int:
        """Return the acting player's choice of action

        Requires parsing the information state from the game context.

        Args:
            context: Tuple of events that have occured so far
            num_legal_actions: Number of legal actions for acting player
            train: Whether learner is training (or evaluating)
        """
        if len(context) == 2:
            info_state = context[0]
            return self.alice.act(info_state, num_legal_actions, train)
        if len(context) == 3:
            info_state = sad_transform(self.alice, context, self.sad)
            return self.bob.act(info_state, num_legal_actions, train)
        raise (Exception)

    def update_rates(self) -> None:
        """Update learning and exploration rates for learners"""
        self.alice.update_rates()
        self.bob.update_rates()

    @abstractmethod
    def update_from_episode(self, episode: list) -> None:
        """Update learners from episode

        Args:
            episode: List of cards, actions and the payoffs
        """


class IndependentQLearner(MultiAgentLearner[QLearner]):
    def __init__(
        self, alice: QLearner, bob: QLearner, sad: Optional[Sads], avd: bool,
    ) -> None:
        """Independent Q-learning

        Args:
            See base class
            avd: Whether to use additive value decomposition (aka VDN)

        Attributes:
            See base class
            avd (bool): Whether to use additive value decomposition (aka VDN)
        """
        super().__init__(alice, bob, sad)
        self.avd = avd

    def update_from_episode(self, episode: list) -> None:
        """Update both learners from episode

        Args:
            See base class
        """
        if self.avd:
            c1, c2, a1, a2, payoff = episode
            alice_is_ = (c1, a1)
            bob_is_ = sad_transform(self.alice, tuple(episode), self.sad)
            # AVD requires all agents to act at every time step. For turn-based
            # games, this means that the non-acting agents must take a "no-op"
            # action (ie one that doesn't do anything). Alice/Bob did not see
            # the infostates at which they had no-ops during the episode so we
            # add them here.
            self.bob.actor.params.add_state(c2, 1)
            self.alice.actor.params.add_state(alice_is_, 1)
            avd_update(
                self.alice, self.bob, c1, c2, a1, 0, 0, alice_is_, bob_is_, False
            )
            avd_update(
                self.alice,
                self.bob,
                alice_is_,
                bob_is_,
                0,
                a2,
                payoff,
                None,
                None,
                True,
            )
        else:
            independent_updates_from_episode(self.alice, self.bob, self.sad, episode)


class IndependentReinforceLearner(MultiAgentLearner[ReinforceLearner]):
    """Independent reinforce learning

    Args:
        See base class

    Attributes:
        See base class
    """

    def __init__(
        self, alice: ReinforceLearner, bob: ReinforceLearner, sad: Optional[Sads],
    ) -> None:
        super().__init__(alice, bob, sad)

    def update_from_episode(self, episode: list) -> None:
        """Update both learners from episode

        Args:
            See base class
        """
        independent_updates_from_episode(self.alice, self.bob, self.sad, episode)


class IndependentA2CLearner(MultiAgentLearner[A2CLearner]):
    def __init__(
        self,
        alice: A2CLearner,
        bob: A2CLearner,
        sad: Optional[Sads],
        use_central_critic: bool,
    ) -> None:
        """Independent advantage actor critic learning

        Alice's critic and Bob's critic must be the same.

        Args:
            See base class
            use_central_critic: Whether to use central critic

        Attributes:
            See base class
            use_central_critic (bool): Whether to use central critic
        """
        super().__init__(alice, bob, sad)
        self.use_central_critic = use_central_critic
        if use_central_critic:
            assert alice.critic is bob.critic
            self.critic = alice.critic

    def update_from_episode(self, episode: list) -> None:
        """Update both learners from episode

        Args:
            See base class
        """
        if self.use_central_critic:
            c1, c2, a1, a2, payoff = episode
            p1_info_state = c1
            p2_info_state = sad_transform(self.alice, tuple(episode), self.sad)
            full_info1 = (c1, c2)
            full_info2 = (c1, c2, a1)
            ac_actor_update(
                self.alice.actor, self.critic, p1_info_state, full_info1, a1
            )
            ac_critic_update(
                self.bob.actor,
                self.critic,
                full_info1,
                a1,
                0,
                full_info2,
                p2_info_state,
            )
            ac_actor_update(self.bob.actor, self.critic, p2_info_state, full_info2, a2)
            self.critic.update_params(full_info2, a2, payoff)
        else:
            independent_updates_from_episode(self.alice, self.bob, self.sad, episode)


def independent_updates_from_episode(
    alice: SingleAgentLearner,
    bob: SingleAgentLearner,
    sad: Optional[Sads],
    episode: list,
) -> None:
    """Update both learners from episode

    Args:
        episode: list cards, actions, and payoffs
    """
    c1, c2, a1, a2, payoff = episode
    p2_info_state = sad_transform(alice, tuple(episode), sad)
    alice.update_from_episode([(c1, a1, payoff)])
    bob.update_from_episode([(p2_info_state, a2, payoff)])


def avd_update(
    alice: QLearner,
    bob: QLearner,
    alice_is: tuple,
    bob_is: tuple,
    alice_a: int,
    bob_a: int,
    r: float,
    alice_is_: Optional[tuple],
    bob_is_: Optional[tuple],
    is_done: bool,
) -> None:
    """Perform additive value decomposition updates on Alice and Bob

    AVD updates a joint Q-function parameterized by additive decomposition
    Q(s_1, s_2, a_1, a_2) = Q(s_1, a_1) + Q(s_2, a2)

    Args:
        alice: Player 1
        bob: Player 2
        alice_is: Alice's information state
        bob_is: Bob's information state
        alice_a: Alice's action
        bob_a: Bob's action
        r: The reward
        alice_is: Alice's next information state
        bob_is_: Bob's next information state
        is_done: Whether the transition was terminal
    """
    alice_q1 = alice.actor.params.vals[alice_is][alice_a]
    bob_q1 = bob.actor.params.vals[bob_is][bob_a]
    if is_done:
        q_next = 0
    else:
        q_next = (
            alice.actor.params.vals[alice_is_].max()
            + bob.actor.params.vals[bob_is_].max()
        )
    alice.actor.params.update_params(alice_is, alice_a, r + q_next - bob_q1)
    bob.actor.params.update_params(bob_is, bob_a, r + q_next - alice_q1)


def sad_transform(
    alice: SingleAgentLearner, context: tuple, sad: Optional[Sads]
) -> tuple:
    """Transform the information state using SAD variant (or don't)

    Args:
        alice: Player 1
        context: The events of the game
        sad: Which simplified action decoding variant to use (or none)

    Returns:
        Player 2's information state
    """
    c1, c2, a1 = context[:3]
    info_state = (c2, a1)
    # If not using sad variant do nothing
    if not sad:
        return info_state
    greedy_a1 = alice.actor.act_greedily(c1)
    explored = a1 != greedy_a1
    signal: Optional[int] = None
    # SAD adds Alice's counterfactual greedy action to Bob's infostate
    if sad.value == "sad":
        signal = greedy_a1
    # Binary SAD adds a boolean indicating whether Alice explored to Bob's
    # information state.
    elif sad.value == "bsad":
        signal = explored
    # Action SAD adds Alice's counterfactual greedy action to Bob's infostate
    # if Alice explored. Otherwise, does nothing.
    elif sad.value == "asad":
        if explored:
            signal = greedy_a1
    # Private SAD adds Alice's private information to Bob's infostate if Alice
    # explored. Otherwise, does nothing.
    elif sad.value == "psad":
        if explored:
            signal = c1
    return (*info_state, signal)
