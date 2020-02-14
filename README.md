[![Build Status](https://travis-ci.com/SwamyDev/udacity-deep-rl-navigation.svg?branch=master)](https://travis-ci.com/SwamyDev/udacity-deep-rl-navigation) [![Coverage Status](https://coveralls.io/repos/github/SwamyDev/udacity-deep-rl-navigation/badge.svg?branch=master)](https://coveralls.io/github/SwamyDev/udacity-deep-rl-navigation?branch=master)
# Udacity Navigation Project

This project is part of the Udacity Reinforcement Learning Nanodegree. In it, a DQN agent is trained to solve an episodic environment where it needs to navigate a 3D world collecting yellow bananas and avoiding blue bananas. The environment is considered solved when the agent receives an average score of >13 over 100 consecutive episodes.

## Environment Setup
### Reward Signal
The agent receives a reward of `+1` when collecting a yellow banana and a reward of `-1` when collecting a blue banana.

### Observation
An observation state consists of the agent's current velocity and ray-based perception of surrounding objects. This state is encoded in a 1x37 tensor.

### Actions
The agent can take 4 discrete actions:
 - `0` move forward
 - `1` move backward
 - `2` turn left
 - `3` turn right
 
## Installation
This project uses `GNU Make` to set up virtual environments and download dependencies. It requires a Linux environment. Under Ubuntu make is part of the `build-essential` package (`apt install build-essential`). Other dependencies are python3 virutalenv (`apt install python3-venv`) and pip (`apt install python3-pip`).


### Setup & Test
To create Python virtual environments and install dependencies run:
```bash
make setup
```

To run all automated tests run:
```bash
make test
```

## Quick Start

### The Command Line Interface
When the environment is set up you can activate the environment (i.e. `source venv/bin/activate`) and you have access to the p1_navigation command-line interface.


Showing help messages:
```bash
p1_navigation --help
```

Training on the Banana environment with the standard configuration:
```bash
p1_navigation -e resources/Banana_Linux/Banana.x86_64 train 3000 -c configs/standard.json
```
The `-e` flag specifies the environment (here the Unity-environment executable). The argument after `train` is the number of episodes the agent should be trained. The -c flag sets the config file to be used for the agent.

### Agent Configuration
The JSON files found in `configs` contain the description of the agents model, learning parameters and epsilon behaviour. The various configs have been tried to find good solutions to the environment and serve as living documentation of the process.

### Running a Saved Agent
The agent is serialized, after a successful training run (by default under `/tmp/p1_navigation_ckpt`). This agent can be loaded and run on an environment. By default the run is rendered and the user can observe the agent interacting with the environment:

```bash
p1_navigation -e resources/Banana_Linux/Banana.x86_64 run /tmp/p1_navigation_ckpt 100
```
The `-e` flag specifies the environment (here the Unity-environment executable). The path after `run` specifies the agent to be loaded. The number after that is the number of episodes the agent should be run.

