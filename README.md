# tiny_hanabi
This package implements the Tiny Hanabi Suite, a collection of six very small two-player common-payoff games, and tabular algorithms that compute joint policies for these games.
The games in the Tiny Hanabi Suite were constructed by Neil Burch and Nolan Bard in 2018.
Each game in the suite is structured as follows:
1) Each player is privately dealt one of `num_cards` cards at uniform random
    (with replacement).
2) Player 1 selects an action, which is publicly observable.
3) Player 2 selects an action.

The payoff of the game is determined by player 1's card, player 2's card,
player 1's action, and player 2's action.

## Installation
- Clone the repository using the command `git clone https://github.com/ssokota/tiny-hanabi.git`
- Enter the directory using `cd tiny-hanabi`
- Install the package with the command `pip install .`

## Getting Started
The file `scripts/interface.py` demonstrates how to use the package.
- Try running `python scripts/interface.py A decpomdp ql --ql_init_lr 0.1 --init_epsilon 0.1 --num_episodes 10000 --plot` to run independent Q-learning on game A for 10,000 episodes.
- Run `open results/figures/example.pdf` to see how it did.

## What's in the Package?
The package includes
1. Tiny Hanabi games A, B, C, D, E, and F.
2. Implementations of each game as a Dec-POMDP, a PuB-MDP, a TB-MDP, and a VB-MDP.
3. Implementations of Q-learning, hysteretic Q-learning, reinforce, and advantage actor critic.
4. Implementations of each single-agent learning algorithm playing independently with itself.
5. Implementations of simplified action decoding (and variations), additive value decomposition (aka VDN), and centralized value functions.

## References

Includes experimental results for algorithms in this package on the Tiny Hanabi Suite:
```
@inproceedings{capi_paper,
title       = {Solving Common-Payoff Games with Approximate Policy Iteration},
journal     = {Proceedings of the AAAI Conference on Artificial Intelligence},
author      = {Sokota, Samuel and Lockhart, Edward and Timbers, Finbarr and Davoodi, Elnaz and D’Orazio, Ryan and Burch, Neil and Schmid, Martin and Bowling, Michael and Lanctot, Marc},
year        = {2021}
}
```


Includes self-contained descriptions of Dec-POMDPs, PuB-MDPs, TB-MDPs, VB-MDPs, simplified action decoding (and variations), additive value decomposition, and centralized value functions:
```
@mastersthesis{capi_thesis,
author       = {Samuel Sokota},
title        = {Solving Common-Payoff Games with Approximate Policy Iteration},
school       = {University of Alberta},
year         = {2020},
}
```


Source for PuB-MDP, TB-MDP:
```
@article{Nayyar2013,
title   = {Decentralized Stochastic Control with Partial History Sharing: A Common Information Approach},
author  = {A. {Nayyar} and A. {Mahajan} and D. {Teneketzis}},
journal = {IEEE Transactions on Automatic Control},
year    = {2013}
}
```

Source for hysteretic Q-learning:
```
@inproceedings{Matignon2007,
author      = {L. {Matignon} and G. J. {Laurent} and N. {Le Fort-Piat}},
booktitle   = {2007 IEEE/RSJ International Conference on Intelligent Robots and Systems},
title       = {Hysteretic Q-learning : an algorithm for Decentralized Reinforcement Learning in Cooperative Multi-Agent Teams},
year        = {2007},
}
```

Source for simplified action decoding:
```
@inproceedings{Hu2020,
title       = {Simplified Action Decoder for Deep Multi-Agent Reinforcement Learning},
author      = {Hengyuan Hu and Jakob N Foerster},
booktitle   = {International Conference on Learning Representations},
year        = {2020},
}
```

Source for additive value decomposition:
```
@inproceedings{Sunehag2018,
author      = {Sunehag, Peter and Lever, Guy and Gruslys, Audrunas and Czarnecki, Wojciech Marian and Zambaldi, Vinicius and Jaderberg, Max and Lanctot, Marc and Sonnerat, Nicolas and Leibo, Joel Z. and Tuyls, Karl and Graepel, Thore},
title       = {Value-Decomposition Networks For Cooperative Multi-Agent Learning Based On Team Reward},
year        = {2018},
booktitle   = {Proceedings of the 17th International Conference on Autonomous Agents and MultiAgent Systems},
}
```

Source for centralized value functions:
```
@inproceedings{Lowe2017,
 author     = {Lowe, Ryan and WU, YI and Tamar, Aviv and Harb, Jean and Pieter Abbeel, OpenAI and Mordatch, Igor},
 booktitle  = {Advances in Neural Information Processing Systems},
 title      = {Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments},
 year       = {2017}
}
```

For an introduction to decentralized Markov decision processes:
```
@book{Oliehoek2016,
author  = {Oliehoek, Frans A. and Amato, Christopher},
title   = {A Concise Introduction to Decentralized POMDPs},
year    = {2016},
}
```

For an introduction to reinforcement learning, Q-learning, reinforce, advantage actor critic:
```
@book{Sutton2018,
author  = {Sutton, Richard S. and Barto, Andrew G.},
title   = {Reinforcement Learning: An Introduction},
year    = {2018 }
}
```
