[![Build Status](https://travis-ci.com/SwamyDev/udacity-deep-rl-navigation.svg?branch=master)](https://travis-ci.com/SwamyDev/udacity-deep-rl-navigation) [![Coverage Status](https://coveralls.io/repos/github/SwamyDev/udacity-deep-rl-navigation/badge.svg?branch=master)](https://coveralls.io/github/SwamyDev/udacity-deep-rl-navigation?branch=master)
# Udacity Projects

This repository is part of the Udacity Reinforcement Learning Nanodegree. It contains solutions to the courses class projects `navigation` and `continuous control`. You can find more detailed explanations for each project and their environments in their dedicated README or Report.md files:

- [Project Navigation](doc/README_p1_navigation.md)
- [Project Continuous Control](doc/README_p2_continuous.md) 
 
## Installation
To run the code of the projects you need to install the repositories virtual environment. To make this as easy as possible it uses `GNU Make` to set up virtual environments and download dependencies. It requires a Linux environment. Under Ubuntu make is part of the `build-essential` package (`apt install build-essential`). Other dependencies are python3 virutalenv (`apt install python3-venv`) and pip (`apt install python3-pip`).

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
This section show exemplary usage of the `udacity-rl` command line interface by training and running agent for the `navigation` project and exploring its environment.

### The Command Line Interface
When the environment is set up you can activate the environment (i.e. `source venv/bin/activate`) and you have access to the udacity-rl command-line interface. With this interface you can run the code for each project. The following section describes how to run a project and how to get help.

Showing help messages:
```bash
udacity-rl --help
```

Example: Training the `navigation` agent on the Banana environment with the standard configuration:
```bash
udacity-rl -e resources/environments/Banana_Linux/Banana.x86_64 train DQN 3000 -c configs/standard.json
```
The `-e` flag specifies the environment (here the Unity-environment executable). The arguments after `train` specify the algorithm to be used and the number of episodes the agent should be trained. The -c flag sets the config file to be used for the agent.

### Agent Configuration
The JSON files found in `configs` contain the description of the agents model(s), learning parameters and epsilon behaviour. The various configs have been tried to find good solutions to the environment and serve as living documentation of the process.

### Running a Saved Agent
The agent is serialized, after a successful training run (by default under `/tmp/agent_ckpt`). This agent can be loaded and run on an environment. By default the run is rendered and the user can observe the agent interacting with the environment:

```bash
udacity-rl -e resources/environments/Banana_Linux/Banana.x86_64 run /tmp/agent_ckpt 1
```
The `-e` flag specifies the environment (here the Unity-environment executable). The path after `run` specifies the agent to be loaded. The number after that is the number of episodes the agent should be run.

### Exploring an Environment
The CLI also allows you to initially explore an environment with an random agent. The environment is rendered by default so you can get a better feel for what is required.

```bash
udacity-rl -e resources/environments/Banana_Linux/Banana.x86_64 explore
```