# UC Berkeley CS285 Deep Reinforcement Learning  Fall 2022
My Solutions of Assignments for [Berkeley CS 285: Deep Reinforcement Learning, Decision Making, and Control](http://rail.eecs.berkeley.edu/deeprlcourse/).

Disclaimer: My solutions did pass all the Gradescope tests but they may still contain errors. Also, don't copy code directly and the solutions here are meant to help you if you get trapped.

## Getting Started
```
# requirements Python=3.7

mujoco==2.2.0
gym==0.25.2
tensorboard==2.10.0
tensorboardX==2.5.1
matplotlib==3.5.3
ipython==7.34.0
moviepy==1.0.3
pyvirtualdisplay==3.0
torch==1.12.1
opencv-python==4.6.0.66
ipdb==0.13.9
swig==4.0.2
box2d-py==2.3.8
networkx==2.5
```
## Folder Structure
Only necessary files are displayed.
```
- hw[number]
    + cs285 (contains running scripts)
        + agents
        + policies
        + ...
    + report.pdf
    + requirement.txt
    + setup.py
    + cs285_hw[num].pdf (hw instructions)
    
- lectures
```

## Homework Topic Summary
```
- HW 1: Imitation Learning
    + Behavioral Cloning (BC)
    + DAgger

- HW 2: Policy Gradients
    + Implementing Policy Gradients
    + Small-Scale Experiments
    + Implementing Neural Network Baselines
    + Implementing Generalized Advantage Estimation

- HW 3: Q-Learning and Actor-Critic Algorithms
    - Q-Learning
        + basic Q-learning performance (DQN)
        + double Q-learning (DDQN)
        + experimenting with hyperparameters
    - Actor-Critic
    - Soft Actor-Critic (SAC)

- HW 4: Model-Based Reinforcement Learning
    + Dynamics Model (Dyna)
    + Action Selection using MBRL / CEM
    + On-Policy Data Collection
    + Ensembles (MBPO)

- HW 5: Exploration Strategies and Offline Reinforcement Learning
    + Random Network Distillation (RND) Algorithm
    + Boltzman Exploration
    + Conservative Q-Learning (CQL) Algorithm
    + Advantage Weighted Actor Critic (AWAC) Algorithm
    + Implicit Q-Learning (IQL) Algorithm

```

