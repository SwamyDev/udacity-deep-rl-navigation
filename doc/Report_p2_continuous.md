# Project: Continuous Control

In this report, I'll show how the Deep Deterministic Policy Gradient (`DDPG`) algorithm is applied to solve a continuous control task. The difference to a discrete action task is, that the agent's action space is continuous and therefore infinite. Usually, this action is represented as a tensor of floating-point values. The infinite action space of continuous control takes, makes them challenging to solve with Reinforcement Learning algorithms. The `DDPG` algorithm has been specifically designed to address this problem. In the concrete case of this project, the agent has to control a simulated robotic arm to reach within a moving target area which is rotating around the arm. The environment was again created using Unity. For details on the environment have a look at [README_p2_continuous.md](README_p2_continuous.md). As with the last project I was able to implement the agent relatively quick, and without running into too many defects, by using best software engineering practises such as Continuous Integration (`CI`) and Test Driven Development (`TDD`).

## Deep Deterministic Policy Gradient
Similar to `Actor-Critic` methods the `DDPG` learns two functions, an action-value function `Q` and a policy `mu` (the Greek letter `mu` is used instead of `pi` to indicate that this is a deterministic policy). 

In concept `DDPG` works similar to `DQN` as the goal is to estimate the optimal action-value function `Q*`. However, continuous control tasks pose a challenge in this regard. Specifically the calculation of the maximum value of `Q*`. Because there are a potentially infinite number of actions, it is hard to find the action with the maximum value. That is where the policy `mu` comes into play.

The deterministic policy `mu` is used to approximate the maximum action of `Q(s, a)` via `Q(s, mu(a))`. Intuitively one can think of `mu` trying to optimize the continuous actions to maximize Q. Other than that, the `DDPG` works similar to `DQN` and can also make use of the algorithmic improvements used in `DQN` such as replay buffers and soft updating target models.

### The DDPG Algorithm
The algorithm implemented in this project works as follows:

1) Select an action from the policy model `mu_local(s)` and add some scaled uniform random noise to it
    - Adding noise ensures sufficient exploration early on and the magnitude of the noise is reduced as training progress. This is similar to the epsilon greedy strategy of `DQN`. 
    - According to more recent findings suggest that Ornstein–Uhlenbeck noise is not strictly necessary, hence I've used this much simpler process
2) Advance the environment with the selected action and record the trajectory (current state; next state; if its the terminal state; reward) in the memory buffer
    - Recording and storing trajectories allows neural networks to be trained efficiently in batches
    - Keeping records of trajectories also allows them to be used multiple times in training
3) Sample trajectory batches randomly from the memory buffer
    - By sampling randomly from the buffer trajectories are decorrelated from each other improving training
4) Use the target policy `mu_target(s_next)` to get the estimated best actions for the next state `a_next`
5) Use the target critic `Q_target(s_next, a_next)` to get the Q values of the next state `Q_next` and estimated best action `a_next`
    - `Q_next` is zero in case the next state is the terminal state 
6) Calculate the current target action value: `Q_target = reward + gamma * Q_next`
    - Notice the missing `max` function - the maximum action estimate is already encoded in `mu_target`
7) Fit the local critic `Q_local(s, a)` to the calculated `Q_target`
8) Fit the local actor `mu(s)` to _maximize the Q-value_ of the local critic using gradient ascent (ascent(`Q_local(s, mu(s))`))
    - In practice, gradient descent is used and the sign of the Q-value produced by the critic flipped because common algorithm frameworks such as `PyTorch` implement only gradient descent.
9) Soft update the target models of `actor` and `critic` by linear interpolation between `Q_target` and `Q_local` respectively `mu_target` and `mu_local`
    - This stabilizes the algorithm, as the model used for training is updated at a different velocity as the one used for predicting and exploring the environment (see the [carrot stick riding](https://www.youtube.com/watch?v=-PVFBGN_zoM) analogy).

## Notes on Development
Again, using good software practises such as `TDD` helped me tremendously to implement the algorithms quickly. By catching bugs early and making sure all the supporting code such as the memory replay buffer is working as I would expect them to, I could focus on getting the algorithm itself right. RL algorithms also tend to work even if there is a subtle bug. By rigorously testing my code I reduced the chance that subtle bugs sneak into my algorithm and unexpectedly crash performance. 

To make sure each iteration I do on the algorithm is properly tested, I need a high test frequency which means I need fast testing times. This is difficult when dealing with RL algorithms as they can often take hours to train - especially on the relatively weak machines usually used in `CI`. Like in the `navigation` project I used specially designed environments to handle this issue. However, I hadn't developed a testing environment for continuous control tasks yet. So I designed a new environment for my [quickcheck](https://github.com/SwamyDev/gym-quickcheck) OpenAI gym extension.

I came up with an environment called `n-knob` where initially 3 random floating-point values are sampled. The goal of the agent is to get as close to these values as possible within 10 steps. The observation tensor of the environment indicates the direction the current guess of the agent needs to change to get closer to the goal. A random agent performs poorly on these tasks, while a well-implemented agent can learn to get a higher reward relatively quickly. However, this environment still needs some work. Training still takes too long, and the difference between random and trained agent is quite narrow. Different observations or reward signals might improve this environment.

Again I use code metrics such as code coverage and `CI` to ensure code quality and find bugs as early as possible. This helps a lot in increasing my development speed. Also being able to access code from the previous `navigation` project such as the memory replay buffer sped up development. Another benefit of having good code coverage is, that it is easy to adapt and generalise code from the `navigation` project and use it for `continuous control`. With my automated tests in place, I can be confident, that the code is still working after implementing any changes.

I've also improved logging a bit, by using the python logging package properly. This was mainly done to avoid spamming my automated test logs with logs from my implementation.
 
Finally, I've again been using config files for controlling all agent hyperparameters. These files can be checked into source control, and with proper commit messages can serve as nice process documentation.

## Results
Using a neural network architecture similar to the one used in the [bipedal example](https://github.com/udacity/deep-reinforcement-learning/blob/master/ddpg-bipedal/model.py), my agent solved the environment after about ~150 episodes. 

![Graph of Training Run](../resources/images/ddpg_training.png)

The final agent used the `ddpg_ann_a_2x256_c_2x256_1x128-2020-02-21` agent configuration:
```json
{
  "eps_start": 1,
  "eps_end": 0.0001,
  "eps_decay": 0.995,
  "batch_size": 128,
  "record_size": 1000000,
  "actor": {
    "layers": [
      {
        "activation": "relu",
        "size": 256
      },
      {
        "activation": "relu",
        "size": 256
      }
    ],
    "device": "cuda:0",
    "lr": 1e-4
  },
  "critic": {
    "layers": [
      {
        "activation": "leaky_relu",
        "size": 256
      },
      {
        "activation": "leaky_relu",
        "size": 256
      },
      {
        "activation": "leaky_relu",
        "size": 128
      }
    ],
    "device": "cuda:0",
    "lr": 3e-4
  },
  "gamma": 0.99,
  "tau": 1e-3
}
```
(tau specifies the linear interpolation value used for the "soft update" of target and local models)

The final trained model is stored under `resources/models/p2_reacher_final`.

## Future Work
The agent seems to work quite well, but it still sometimes misses the goal. Maybe a different reward function based on the distance to the target would improve performance. With that, the agent could better gauge its performance. I've observed a similar issue when implementing the `n-knob` environment. When first implementing the environment I used a fixed numerical reward. However, it proved to be too hard for an agent to learn a proper policy from that because it never knew if it was doing any better by trying different actions. Hitting the right values was simply to rare to change the reward signal and subsequently for the agent to learn a good policy. Changing the reward signal to be based on the L2 norm distance to the target values proved to be essential.

Additionally, I could also implement more recent improvements to the `DDPG` algorithm to improve performance, such as `Twin-Delayed DDPG` or `Soft Actor-Critic`. Both improvements address an issue with `DDPG` where the Q-value function tends to severely overestimate the value of state-action pairs. This overestimation is then exploited by the policy network and leads to a break-down in the performance of the algorithm. This makes the `DDPG` very sensitive to the choice of hyperparameters and quite brittle in general. The `Soft Actor-Critic` also improves the exploration/exploitation trade-off by improving the action selection mechanism. However, the final agent is already quite robust and learns a good policy reliably. 

Again I haven't done any extensive hyperparameter search, and I'm convinced that with a simple random search one could improve the current result.
