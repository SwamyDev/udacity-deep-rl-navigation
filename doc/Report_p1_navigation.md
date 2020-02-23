# Project: Navigation Report

This report illustrates, how Deep Q-Networks are applied to solve a gathering task within a 3D environment. The environment was created using Unity and the goal is to collect as many yellow bananas as possible while avoiding the blue ones. See the [README_p1_navigation.md](README_p1_navigation.md) for a more detailed description of the environment. In this report, I also try to highlight the importance of diligent software practises and how they played a crucial role in developing a working DQN agent.

## Deep Q-Networks
In traditional Q-Learning the agent tries to find the optimal policy (`pi*`) by trying to estimate the optimal action-value function `Q*` which maps state-action pairs to their value. The value of a state-action pair is the expected (discounted) accumulated return of the agent when it follows its policy. If the agent has access to `Q*` getting to `pi*` is trivial, as the agent simply needs to select the action with the highest value. However, computing `Q*` is the tricky part.

### The DQN Algorithm
The algorithm implemented in this project works as follows:

1) Epsilon-greedily select an action from the target policy model `theta_target`
    - An epsilon greedy strategy is used to ensure sufficient exploration of the environment
2) Advance the environment with the selected action and record the trajectory (current state; next state; if its the terminal state; reward) in the memory buffer
    - The memory buffer is needed, to make the training of neural networks used for Q-value estimation more stable and efficient
3) Sample trajectory batches from the memory buffer
4) Use the target model to get the action values of the next state `Q_next`
    - `Q_next` is zero in case the next state is the terminal state
5) Calculate the current target action value: `Q_target = reward + gamma * max(Q_next)`
6) Fit the local model `theta_local` to the current `Q_target`
7) Soft update the target model by linear interpolation between `theta_target` and `theta_local`
    - This also stabilizes the algorithm, as the model used for training is updated at a different velocity as the one used for predicting and exploring the environment (see the [carrot stick riding](https://www.youtube.com/watch?v=-PVFBGN_zoM) analogy).

## Notes on Development
One of the most difficult issues when implementing reinforcement learning algorithms are subtle to catch bugs. The problem is, that because of the nature of these algorithms, they sometimes seem to learn something even if the code contains defects. To counter this issue I've employed a test-driven development approach. It is a process that has been already used successfully in other software development projects and it seems to work equally well for developing RL algorithms. By stating my assumption what the algorithm should do as a failing test first, I can make sure that once the test passes, that this part is implemeted. I also can be sure, that the test would catch it if it didn't work properly. I also found it helpful to derive the algorithm thinking in terms of "which bit should not yet work". This helped me get there in small increments.

To accomplish this and to be able to use it in a low spec CI environment I've developed the custom gym environment `quickcheck`. It, for instance, contains the `RandomWalk` environment as described in Sutton's and Barto's book "Reinforcement Learning: An Introduction". This simple environment is sufficient to catch most bugs quickly, and it can verify that my agent is learning something. 

Furthermore, I make extensive use of code coverage metrics to make sure that all my processes are tested. Especially in parts of the code base where I haven't been as diligent with the TDD process. With coverage, however, I can find blind spots and fix them. 

Additionally I noticed that it is vital to log as much information about the training process as possible. This helps to confirm or disprove certain assumptions about the training process when things do not go as expected.

I've been using config files for controlling all agent hyperparameters. These files can be checked into source control, and with proper commit messages can serve as nice process documentation.

## Results
Using a very simple neural network architecture with just a very small hidden layer of 16 neurons it is possible to solve the Banana environment. The final agent that I've trained solved it after ~700 episodes.

![Graph of Training Run](../resources/images/dqn_training.png)

The final model used the `ann_1x16_20-14-02.json` agent configuration:
```json
{
  "eps_start": 1,
  "eps_end": 0.01,
  "eps_decay": 0.995,
  "batch_size": 64,  
  "record_size": 100000,
  "model": {
    "layers": [  
      {
        "activation": "relu",
        "size": 16
      }
    ],
    "device": "cpu",
    "lr": 0.0005  
  },
  "gamma": 0.99,
  "tau": 0.001  
}
```
(tau specifies the linear interpolation value used for the "soft update" of target and local models)

What's interesting is that the agent successfully solves the environment so quickly given such a small neural network.

The final trained model is stored under `resources/models/p1_navigation_final`.

## Future Work
I haven't done any extensive hyperparameter search, and I'm convinced that with a simple random search one could improve the current result.

Of course, one could extend the DQN algorithm and implement things like Double DQN, prioritized experience replay, etc. to improve the algorithm.

One of the main issues I still have with the code is that currently I simply print the logs to the standard out. Given the importance of good logs for developing RL algorithms, a proper logging system would be a good idea. 

