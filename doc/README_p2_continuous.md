# Project: Continuous Control

This project is part of the Udacity Reinforcement Learning Nanodegree. In this project a `DDPG` agent is trained to solve a continuous control task. Specifically the agent needs to control a robot arm to reach a target rotating around the arm. The agent receives a reward for each time step the arm is within the target location. The environment is considered solved when the agents scores an average of >30 points over the course of 100 episodes.

## Environment Setup
### Reward Signal
The agent receives a reward of `0.1` each time step the robot arm is within the target area. The goal for the agent is therefore to stay as long within the target area as possible.

### Observation
An observation state consists of the agent's current position, rotation, velocity and angular velocities of the arm. This state is encoded in a 1x33 tensor.

### Actions
The action the agent can take consists of a 1x4 tensor corresponding to the torque applicable to the two joints of the arm. The torque value of the action tensor is normalized to a range between `-1` and `1`.
 
## Exploring
To explore the `Reacher_Linux` environment run the following command from the root of the repository:
```bash
udacity-rl -e resources/environments/Reacher_Linux/Reacher.x86_64 explore
```
 
## Training
To train the `continuous control` agent run the following command from the root of the repository:
```bash
udacity-rl -e resources/environments/Reacher_Linux/Reacher.x86_64 train DDPG 500 -c configs/ddpg_ann_a_2x256_c_2x256_1x128-2020-02-21.json
```

## Running
To observe the stored final agent run the following command from the root of the repository:
```bash
udacity-rl -e resources/environments/Reacher_Linux/Reacher.x86_64 run resources/models/p2_reacher_final/ 1
```
