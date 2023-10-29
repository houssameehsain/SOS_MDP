# SOS_MDP
 Deep reinforcement learning to solve self-organized settlement growth Markov decision processes. 

## About this repository
This repository contains source files for my MS thesis titled, "Deep reinforcement learning for urban modeling: A test case on self-organized settlement morphogenesis simulation", at Bilkent University.
The repository comprises the training and hyperparameter tuning code for training RL agents using the DQN and PPO algorithms to generate self-organized settlement formations under the SOS-MDP framework. In addition, implementations of the RL environments within Rhino/Grasshopper are available in the `RhGh_envs` directory. The results of the training of RL agents for the research experiments are available in the `logs` directory, and the hyperparameter tuning logs are found within the `hyperparameter_tuning` directory. 

All the RL environments found in the `envs.py` script are compatible with the OpenAI Gym convention, which is widely used within the RL community. This will ease the use of the implementation of the Beady Ring and Path Node environments in future research. The implementations could also be easily extended to incorporate more performance metrics and urban growth dynamics. The OpenAI Gym structure allows for training agents in these environments using various RL algorithms. 

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
