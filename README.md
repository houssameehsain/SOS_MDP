# SOS_MDP
 Deep reinforcement learning to solve self-organized settlement growth Markov decision processes. 
<p float="left">
  <img src="logs/gpnv0-ppo-episode-01.gif" width="150" />
  <img src="logs/gpnv0-ppo-episode-02.gif" width="150" /> 
  <img src="logs/gpnv0-ppo-episode-03.gif" width="150" />
</p>

## About this repository
This repository contains source files for my MS thesis titled, "Deep reinforcement learning for urban modeling: A test case on self-organized settlement morphogenesis simulation", at Bilkent University.
The repository comprises the training and hyperparameter tuning code for training RL agents using the DQN and PPO algorithms to generate self-organized settlement formations under the SOS-MDP framework. In addition, implementations of the RL environments within Rhino/Grasshopper are available in the `RhGh_envs` directory. The results of the training of RL agents for the research experiments are available in the `logs` directory, and the hyperparameter tuning logs are found within the `hyperparameter_tuning` directory. 

All the RL environments found in the `envs.py` script are compatible with the OpenAI Gym convention, which is widely used within the RL community. This will ease the use of the implementation of the Beady Ring and Path Node environments in future research. The implementations could also be easily extended to incorporate more performance metrics and urban growth dynamics. The OpenAI Gym structure allows for training agents in these environments using various RL algorithms. 

## Hyperparameters
The following two tables list the best hyperparameters found during the tuning process for each environment that were used for training both the DQN and PPO agents. 

| DQN hyperparameters	| GBR v1 (CNN)	| GBR v1 (Moore)	| GBR v2	| GPN |
| :---         |     :---:      |     :---:      |     :---:      |     :---:      |
| Batch size | 256 |	128 |	256 |	64 |
| Buffer size |	10k	| 50k |	50k |	100k |
| Exploration final rate |	3.76e-02 |	1.04e-03	| 4.48e-03 |	1.12e-02 |
| Exploration fraction	| 1.3e-01	| 3.9e-02	| 6.91e-03	| 3.36e-03 |
| Gamma |	0.999 |	0.999 |	0.99	| 0.99 |
| Learning rate |	6.21e-04 |	8.09e-04 |	1.02e-04 |	3.99e-02 |
| Learning starts	| 5000	| 0	| 5000	| 10k |
| Number of layers |	3 |	1 |	1 |	2 |
| Layer size |	256	| 64	| 64	| 64 |
| Gradient steps	| 256	| 4	| 128	| 16 |
| Target update interval	| 5000	| 1	| 20k	| 15k |

Table 1 Hyperparameters used to train DQN agents in each environment. Moore refers to the local observation version of the GBR v1, while CNN means Convolutional neural network, referring to the global observation version of that same environment.

| PPO hyperparameters |	GBR v1	| GBR v2	| GPN |
| :---         |     :---:      |     :---:      |     :---:      |
| Activation function	| ReLU	| Tanh	| Tanh |
| Batch size	| 256	| 64	| 64 |
| Clip range	| 0.3	| 0.4	| 0.4 |
| Entropy coefficient	| 2.52e-07	| 4.03e-08	| 2.99e-07 |
| GAE lambda	| 0.99	| 0.98	| 0.92 |
| Gamma	| 0.99	| 0.995	| 0.999 |
| Learning rate	| 1.26e-05	| 1.21e-04	| 1.04e-04 |
| Max. gradient norm	| 0.9	| 0.9	| 0.3 |
| Number of epochs	| 20	| 10	| 20 |
| Number of steps	| 2048	| 512	| 256 |
| Number of layers	| 2	| 3	| 2 |
| Layer size	| 256	| 256	| 256 |
| Orthogonal Initialization	| Yes	| No	| Yes |
| Value function coefficient	| 2.69e-06	| 4.97e-04	| 9.5e-04 |

Table 2 Hyperparameters used to train PPO agents in each environment.

## Perlin Simplex Noise Parameters 
To reproduce the topographic conditions (introvert, linear, extrovert) used within the GPN environment, the parameters to generate those conditions are as follows:
| Perlin noise parameters	| Introvert	| Linear	| Extrovert |
| :---         |     :---:      |     :---:      |     :---:      |
| Starting location	| 50, 50 |	90, 90	| 50, 40 |
| Scale	| 30	| 40	| 50 |
| Octaves	| 3	| 2	| 2 |
| Persistence	| 0.4	| 0.4	| 0.4 |
| Lacunarity	| 2.0	| 2.0	| 1.0 |
| Central flatness	| 0.5	| 0.5	| 0.5 |
| Frequency	| 1	| 1	| 1 |
| Amplitude	| 1	| 1	| 1 |
| Initial value	| 0	| 0	| 1 |
| Random seed	| 77	| 95	| 123 |

Table 3 Perlin Simplex noise parameters used to generate the topographic conditions within the GPN environment.

## Dependencies
We make use of Stable-Baselines3 (Raffin et al., 2021) implementations of the various RL algorithms (DQN, PPO, A2C) to ensure stable training results as well as the reproducibility of those results by others. The hyperparameter tuning process is performed using the Optuna (Akiba et al., 2019) implementation of Bayesian search over a predefined hyperparameter space. Make sure to install all the dependencies from the `requirements.txt` file.

## Abstract
Self-organized modes of urban growth could result in high-quality urban space and have notable benefits such as providing affordable housing, and thus, wider access to economic opportunities within cities. Modeling this non-linear, complex and dynamic sequential urban aggregation process requires adaptive sequential decision-making. In this study, a deep reinforcement learning (DRL) approach is proposed to automatically learn these adaptive decision policies to generate self-organized settlements that maximize a certain performance objective. A framework to formulate the self-organized settlement morphogenesis problem as single-agent reinforcement learning (RL) environment is presented. This framework is then verified by developing three environments based on two cellular automata urban growth models and training RL agents using the Deep Q-learning (DQN) and Proximal Policy Optimization (PPO) algorithms to learn sequential urban aggregation policies that maximize performance metrics within those environments. The agents consistently learn to sequentially grow the settlements while adapting their morphology to maximize performance, maintain right-of-way, and adapt to topographic constraints. The method proposed in this study can be used not only to model self-organized settlement growth based on preset performance objectives but also could be generalized to solve various single-agent sequential decision-making generative design problems. 

**Keywords:** self-organized settlements, urban morphology, urban modeling and simulation, machine learning, deep reinforcement learning, agent-based modeling, design control and planning

## Citing this work
```bibtex
@thesis{hsain2023sosmdp,
  title={Deep Reinforcement Learning for Urban Modeling: Morphogenesis Simulation of Self-Organized Settlements},
  author={Hsain, Houssame Eddine},
  department={Graduate School of Engineering and Science - Department of Architecture},
  university={Bilkent University},
  location={Ankara, Turkey},
  year={2023}
}
```
