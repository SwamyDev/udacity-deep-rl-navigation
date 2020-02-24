# Project: Navigation

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
 
## Exploring
To explore the `Banana_Linux` environment run the following command from the root of the repository:
```bash
udacity-rl -e resources/environments/Banana_Linux/Banana.x86_64 explore
```
 
## Training
To train the `navigation` agent run the following command from the root of the repository:
```bash
udacity-rl -e resources/environments/Banana_Linux/Banana.x86_64 train DQN 1000 -c configs/ann_1x16_20-14-02.json
```

## Running
To observe the stored final agent run the following command from the root of the repository:
```bash
udacity-rl -e resources/environments/Banana_Linux/Banana.x86_64 run resources/models/p1_navigation_final/ 1
```