"""Learning settings for decentralized control. Most obvious setting is the
DecPOMDP, in which multiple learners interact with the environment. Other
settings are various ways of centralized the learning problem. In these
settings, a single agent dictates the policy of all players. Inheritance
structure is:

Game:
    -> DecPOMDP : For multi-agent learning algorithms
    -> PuBMDP : For single-agent learning algorithms
        (exploits public knowledge)
    -> TBMDP : For single-agent learning algorithms
        (exploits temporal knowledge)
    -> VBMDP : For single-agent learning algorithms
        (exploits vacuous knowledge)
"""

from abc import ABC, abstractmethod
from enum import Enum
import itertools
from typing import Optional, Union, Dict

import numpy as np


statetype = Union[None, int, tuple]


class Settings(Enum):
    decpomdp = "decpomdp"
    pubmdp = "pubmdp"
    tbmdp = "tbmdp"
    vbmdp = "vbmdp"


class Game(ABC):
    def __init__(self, payoffs: np.ndarray, optimal_return: float) -> None:
        """Base class for games

        Args:
            payoffs: Payoff matrix indexed card1, card2, action1, action2
            optimal_return: Expected return of an optimal joint policy

        Attributes:
            num_cards (int): The number of cards in the deck
            num_actions (int): The number of actions available to each player
            history: (list): The events that have occurred so far
                Note that this is NOT a history in a technical sense, it is
                a list of information required for agents to perform updates.
            num_players (int): The number of players in the game. It may either
                be two or one depending on the setting so it is initialized
                to `None` in the base class.
            horizon (int): The length of terminal histories. It varies from
                two to four depending on the setting so it is initialized to
                `None` in the base class.
            payoffs: See args
            optimal_return: See args
        """
        self.num_cards = payoffs.shape[0]
        self.num_actions = payoffs.shape[-1]
        self.reset()
        self.num_players = 0
        self.horizon = 0
        self.payoffs = payoffs
        self.optimal_return = optimal_return

    @abstractmethod
    def start_states(self) -> tuple:
        """Return tuple of start states for the game"""

    def random_start(self) -> None:
        """Initialize history at random start state (each is equiprobable)"""
        start_states = self.start_states()
        self.history = start_states[np.random.choice(range(len(start_states)))]

    def is_terminal(self) -> bool:
        """Checks whether the game is over"""
        return len(self.history) == self.horizon

    def payoff(self) -> float:
        """Return the reward for the current history

        In Tiny Hanabi games the reward is zero unless the history is terminal
        """
        return self.payoffs[tuple(self.history)] if self.is_terminal() else 0.0

    @abstractmethod
    def step(self, action: int) -> None:
        """Process the player's action"""

    @abstractmethod
    def num_legal_actions(self) -> int:
        """Return the number of legal actions at the current history"""

    @abstractmethod
    def context(self) -> statetype:
        """Return the context required by the acting player"""

    @abstractmethod
    def episode(self) -> list:
        """Return information about the current episode's trajectory"""

    def cummulative_reward(self) -> float:
        """Return the cummulative reward for the current episode"""
        return self.payoff()

    def reset(self, history: Optional[list] = None) -> None:
        """Reset the game to `history` or a random start state

        Args:
            history: History to reset the game to; `None` means random start
        """
        if history is None:
            self.random_start()
        else:
            self.history = history


class DecPOMDP(Game):
    def __init__(self, payoffs: np.ndarray, optimal_return: float):
        """Decentralized partially observable Markov decision process

        Args:
            See base class

        Attributes:
            See base class
        """
        super().__init__(payoffs, optimal_return)
        self.num_players = 2
        self.horizon = 4

    def start_states(self) -> tuple:
        """Returns tuple of possible deals [card1, card2]"""
        return tuple(
            [i, j] for i in range(self.num_cards) for j in range(self.num_cards)
        )

    def step(self, action: int) -> None:
        """Appends `action` to `history`"""
        self.history.append(action)

    def num_legal_actions(self) -> int:
        """Return the number of legal actions at the current history"""
        return self.num_actions

    def context(self) -> tuple:
        """Return the history (agent must parse into infostate)"""
        return tuple(self.history)

    def episode(self) -> list:
        """Return the history and the payoff"""
        return self.history + [self.payoff()]


class PuBMDP(Game):
    def __init__(self, payoffs: np.ndarray, optimal_return: float) -> None:
        """Public belief Markov decision process

        Args:
            See base class

        Attributes:
            See base class
            payoffs (dict): The payoffs for the PuB-MDP
            beliefs (dict): Maps a tuple of the first action and first
                prescription to the corresponding public belief state.
        """
        super().__init__(payoffs, optimal_return)
        self.num_players = 1
        self.horizon = 4
        self.build(payoffs)

    def start_states(self) -> tuple:
        """Return the start states

        The start states are the deals to player 1.
        """
        return tuple([c] for c in range(self.num_cards))

    def step(self, prescription: int) -> None:
        """Update the game given the prescription

        For the first time step, the game picks player 1's action according to
        prescription and the deal for player one. It adds the prescription and
        the corresponding belief state to the history. For the second time
        step, the game appends the prescription to the history.

        Args:
            prescription: The coordinator's action
        """
        if len(self.history) == 1:
            prescription1_table = table_repr(
                prescription, self.num_cards, self.num_actions
            )
            action = np.argmax(prescription1_table[self.history[0]])
            belief = self.beliefs[prescription, action]
            self.history += [prescription, belief]
        elif len(self.history) == 3:
            self.history.append(prescription)

    def context(self) -> Optional[int]:
        """Return the public belief state

        Before the coordinator has acted, there is only one public belief
        state since the coordinator has no information. After the coordinator
        has acted once, the public belief becomes the last item in `history`.
        Note that the return values are bijective with the public belief states
        rather than the public belief states themselves.
        """
        return None if len(self.history) == 1 else self.history[-1]

    def episode(self) -> list:
        """Return a list of (state, action, reward) PuB-MDP transitions"""
        return [(None, self.history[1], 0), (*self.history[2:], self.payoff())]

    def num_legal_actions(self) -> int:
        """Return the number of legal prescriptions"""
        return self.num_prescriptions

    def build(self, payoffs: np.ndarray) -> None:
        """Build the PuB-MDP corresponding to `payoffs`

        Build the game by looping over all PuB-MDP trajectories.

        Args:
            payoffs : Payoff matrix indexed card1, card2, action1, action2
        """
        num_possible_info_state = self.num_cards
        self.num_prescriptions = self.num_actions ** num_possible_info_state
        self.beliefs = {}
        self.payoffs = {}
        # First loop over the coordinator's initial prescriptions
        for prescription1 in range(self.num_prescriptions):
            prescription1_table = table_repr(
                prescription1, self.num_cards, self.num_actions
            )
            # Then loop over the private obs for player 1
            for c1 in range(self.num_cards):
                # Public obs corresponding to the private obs `c1`
                a1 = np.argmax(prescription1_table[c1])
                # Posterior over player 1's private obs from coordinator's view
                possible_c1 = np.flatnonzero(prescription1_table[:, a1])
                # Corresponding public belief state
                b = (tuple(possible_c1), a1)
                self.beliefs[prescription1, a1] = b
                for prescription2 in range(self.num_prescriptions):
                    prescription2_table = table_repr(
                        prescription2, self.num_cards, self.num_actions
                    )
                    tmp = []
                    # Payoff in PuB-MDP = expected payoff over public belief
                    for c1_ in possible_c1:
                        for c2_ in range(self.num_cards):
                            a2 = np.argmax(prescription2_table[c2_])
                            tmp.append(payoffs[c1_, c2_, a1, a2])
                    self.payoffs[c1, prescription1, b, prescription2] = np.mean(tmp)


class TBMDP(Game):
    def __init__(self, payoffs: np.ndarray, optimal_return: float) -> None:
        """Temporal belief Markov decision process

        Args:
            See base class

        Attributes:
            See base class
            payoffs (dict): The payoffs for the TB-MDP
            legal_actions (dict): Maps temporal belief state to the number of
                legal prescriptions.
        """
        super().__init__(payoffs, optimal_return)
        self.num_players = 1
        self.horizon = 2
        self.build(payoffs)

    def start_states(self) -> tuple:
        """Return the start states

        The TB-MDP is deterministic so there is only one start state.
        """
        return ([],)

    def step(self, action: int) -> None:
        """Append action to history"""
        self.history.append(action)

    def num_legal_actions(self) -> int:
        """Return the number of legal prescriptions

        The size of the support of the temporal belief state varies based on
        the first prescription. As a result, the number of legal second
        prescriptions depends on the first prescription.
        """
        if len(self.history) == 0:
            return self.legal_actions[None]
        return self.legal_actions[self.history[0]]

    def context(self) -> Optional[int]:
        """Return the temporal belief state

        Note that the return values (nothing on the first time step and the
        first prescription on the second time step) are bijective with the
        temporal belief states rather than the temporal belief states
        themselves.
        """
        if len(self.history) == 0:
            return None
        elif len(self.history) == 1:
            return self.history[0]
        raise (Exception)

    def episode(self) -> list:
        """Return a list of (state, action, reward) TB-MDP transitions"""
        return [
            (None, self.history[0], 0),
            (self.history[0], self.history[1], self.payoff()),
        ]

    def build(self, payoffs: np.ndarray) -> None:
        """Build the TB-MDP corresponding to `payoffs`

        Build the game by looping over all PuB-MDP trajectories.

        Args:
            payoffs : Payoff matrix indexed card1, card2, action1, action2
        """
        self.payoffs = {}
        self.legal_actions: Dict[Optional[int], int] = {}
        num_possible_info_state1 = self.num_cards
        num_prescriptions1 = self.num_actions ** num_possible_info_state1
        self.legal_actions[None] = num_prescriptions1
        # First loop over the coordinator's initial prescriptions
        for prescription1 in range(num_prescriptions1):
            prescription1_table = table_repr(
                prescription1, self.num_cards, self.num_actions
            )
            # Compute possible obs for player 2
            possible_a1 = np.flatnonzero(prescription1_table.max(axis=0))
            # Compute possible info states for player 2
            num_possible_info_state2 = self.num_cards * len(possible_a1)
            # Compute number of legal prescriptions for coordinator
            num_legal_actions2 = self.num_actions ** num_possible_info_state2
            self.legal_actions[prescription1] = num_legal_actions2
            for prescription2 in range(num_legal_actions2):
                prescription2_table = table_repr(
                    prescription2, num_possible_info_state2, self.num_actions
                )
                # Indexing the rows of the second prescription table takes a
                # bit more work because they are indexed by both the player 2's
                # card and player 1's action.
                idx = {
                    (c2_, a1_): i
                    for i, (c2_, a1_) in enumerate(
                        itertools.product(range(self.num_cards), possible_a1)
                    )
                }
                tmp = []
                # Payoff in TB-MDP = expected payoff over temporal belief
                for c1_ in range(self.num_cards):
                    for c2_ in range(self.num_cards):
                        a1_ = np.argmax(prescription1_table[c1_])
                        a2_ = np.argmax(prescription2_table[idx[(c2_, a1_)]])
                        tmp.append(payoffs[c1_, c2_, a1_, a2_])
                self.payoffs[prescription1, prescription2] = np.mean(tmp)


class VBMDP(Game):
    def __init__(self, payoffs: np.ndarray, optimal_return: float) -> None:
        """Vacuous belief Markov decision process

        Args:
            See base class

        Attributes:
            See base class
            payoffs (dict): The payoffs for the VB-MDP
            num_action_profiles (int): The number of action profiles
        """
        super().__init__(payoffs, optimal_return)
        self.num_players = 1
        self.horizon = 1
        self.build(payoffs)

    def start_states(self) -> tuple:
        """Return the start states

        There is only one start state for the VB-MDP.
        """
        return ([],)

    def context(self) -> None:
        """Return the one belief state

        As the name suggests, there is only one belief state in the VB-MDP.
        Note that we are returning something bijective with it, rather than
        actually computing it.
        """
        return None

    def step(self, action: int) -> None:
        """Append `action` to `history`"""
        self.history.append(action)

    def num_legal_actions(self) -> int:
        """Return number of legal prescriptions"""
        return self.num_action_profiles

    def episode(self) -> list:
        """Return list of (start, action, reward) VB-MDP transitions"""
        return [(None, self.history[0], self.payoff())]

    def build(self, payoffs: np.ndarray) -> None:
        """Build the VB-MDP corresponding to `payoffs`

        Build the game by looping over all VB-MDP trajectories.

        Args:
            payoffs: Payoff matrix indexed card1, card2, action1, action2
        """
        self.payoffs = {}
        num_cards = payoffs.shape[0]
        num_actions = payoffs.shape[-1]
        num_p1_info_states = num_cards
        num_p2_info_states = num_cards * num_actions
        num_info_states = num_p1_info_states + num_p2_info_states
        num_p1_action_profiles = num_actions ** num_p1_info_states
        num_p2_action_profiles = num_actions ** num_p2_info_states
        # Number of action profiles
        self.num_action_profiles = num_p1_action_profiles * num_p2_action_profiles
        # Loop over action profiles
        for action_profile in range(self.num_action_profiles):
            action_profile_table = table_repr(
                action_profile, num_info_states, num_actions
            )
            tmp = []
            # Payoff in VB-MDP is expected payoff over the one belief state
            for c1 in range(num_cards):
                for c2 in range(num_cards):
                    a1 = np.argmax(action_profile_table[c1])
                    p2_info_state_idx = num_cards + a1 * num_cards + c2
                    a2 = np.argmax(action_profile_table[p2_info_state_idx])
                    tmp.append(payoffs[c1, c2, a1, a2])
            self.payoffs[(action_profile,)] = np.mean(tmp)


def table_repr(index: int, num_info_states: int, num_actions: int) -> np.ndarray:
    """Express prescription with a table representation.

    Args:
        index: Index of the prescription
        num_info_states: Number of possible info state
        num_actions: Number of legal actions

    Returns:
        `num_states` by `num_actions` representation of prescription where
            (i, j) -> prescribed probabiliy of action j given infostate i.
    """
    table = np.zeros((num_info_states, num_actions))
    for info_state in range(num_info_states):
        table[info_state, index % num_actions] = 1
        index = index // num_actions
    return table
