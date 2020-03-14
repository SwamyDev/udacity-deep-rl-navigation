# Project: Continuous Control

This project is part of the Udacity Reinforcement Learning Nanodegree. In this project, multiple `DDPG` agents are trained to solve a continuous control task. Specifically, each agent needs to control a tennis racket to pass a ball back and forth, keeping it in game as long as possible. The each agent receives a reward of each time it hits the ball over the net and gets penalized when the ball hits the ground or goes out of bounds. Hence, it is in the interest of both agents to keep the ball in play, making this a cooperative environment. The environment is considered solved, when the maximum score of each agent reaches an average of >0.5 points throughout 100 episodes.

## Environment Setup
### Reward Signal
Each agent receives a reward of `0.1` when it hits the ball over the net, but get's a penalty of -0.01 each time the ball hits the ground or goes out of bounds. The goal for both agents is therefore to keep the ball in play as long as possible. 

### Observation
An observation state for each agent individually consists of the agent's current position and velocity and the position and velocity of the ball. The total observation of both agent is encoded in a 2x24 tensor (stacking the observations of both agents). 

### Actions
The action each agent can take consists of a 1x2 tensor corresponding to 2 continuous actions: Moving towards or away from the net, and jumping. The action values are normalized to a range between `-1` and `1`.

## Exploring
To explore the `Tennis_Linux` environment for 100 episodes run the following command from the root of the repository:
```bash
udacity-rl -e resources/environments/Tennis_Linux/Tennis.x86_64 explore -n 100
```
 
## Training
To train the `multi-agent` agents run the following command from the root of the repository:
```bash
udacity-rl -e resources/environments/Tennis_Linux/Tennis.x86_64 train NDDPG 5000 -c configs/multi_ddpg_ann_a_2x256_c_2x256_1x128-2020-03-07.json 
```

## Running
To observe the stored final agent run the following command from the root of the repository:
```bash
udacity-rl -e resources/environments/Tennis_Linux/Tennis.x86_64 run resources/models/p3_tennis_final/ 1
```
