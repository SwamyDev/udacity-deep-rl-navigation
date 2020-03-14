# Project: Continuous Control

This report details how multiple agents using the Deep Deterministic Policy Gradient (`DDPG`) algorithm solve a cooperative task. One of the main issues with multi-agent (`MA`) environments is, that actions of each agent influence observations, rewards and the set of valid actions for each other agent. This makes the problem non-stationary. However, in the particular case of the `Tennis` environment, this issue seems to be not that pronounced. In this environment two agents control a tennis racket and their goal is to keep the ball in play without hitting it out of bounds or dropping it to the ground (more details in [README_p3_multiagent.md](README_p3_multiagent.md)). Each agent can act relatively independent of the actions of the other agent (agents can't obstruct each other for instance) and focus marley on hitting the ball over the net. This makes this environment easier to solve then other `MA` settings, and is possibly the reason why the naive approach of training two independent `DDPG` agents was sufficient. As with the last project I was able to implement the agent relatively quick, and without running into too many defects, by using best software engineering practises such as Continuous Integration (`CI`) and Test Driven Development (`TDD`). Additionally I've learned, that it is a good idea to study the performance of a random agent within the environment, before diving into it. In this particular case it happened quite often that I thought the agent was doing well and getting excited. However, it turned out that there was some bug in the code resulting in constant random exploration. What made this phenomenon more pronounced was the fact that the learning agent would crash initially to the absolute minimum of rewards, hence the random agent actually looked better in comparison. 

## N Deep Deterministic Policy Gradient
The N Deep Deterministic Policy Gradient simply specifies multiple agents using the `DDPG` algorithm independently. For this environment it wasn't necessary to do much tweaking of the original algorithm, such as for learning the `Critic` together. In this implementation I simply reused the memory replay buffer, but without sharing experiences between agents. 

The main "trick" was just to train both agents long enough to get them over the initial crashing phase. Once the algorithm reached peak performance around `>2`the algorithm would still fluctuate a lot, but not crash down to the minimum again. I'm suspecting this is due, the agents learning a careful balance where they would bounce the ball back end forth at the exact same position, for very long periods. This results a very high amount of similar experience leading to oversampling of these states. Intuitively this would mean that the agents do not see many different situations where the ball would come from different angles at different positions and hence are "surprised" when they still happen and do not react properly leading to poor performance until balance is found again.

## The DDPG Algorithm
The algorithm is the same as described in [Report_p2_continuous](Report_p2_continuous.md). However, there are some minor tweaks to the exploration handling. I implement a similar setup as the `DDPG` algorithm described in [Spinning Up RL from OpenAI](https://spinningup.openai.com/en/latest/algorithms/ddpg.html). I now start with completely random "preheating" steps before letting the agent act, and then simply add gaussian noise with a fixed standard deviation to ensure continued exploration. 

## Notes on Development
First I implemented a single `DDPG` agent which would just control both rackets and receive both observations. With this much easier approach to solving the environment I could explore the various pitfalls of it. It turned out in the end that one of the major challenges wasn't agent coordination or the environments being non-stationary, but rather exploration/exploitation trade-off and the reward signal. Even the easier "one-mind" agent would crash early during training and not recover for a long time.

I've spent a lot of time investigating the exploration/exploitation tread-off because this hard initial crash of the agent (constant `-0.05` average reward). The setup described in the previous section turned out to be the best, however, it still didn't improve that much. I tried tuning various other hyper parameters like learning rate, model architecture, gamma and tau. However, none of these improved the performance much. 

I contemplated implementing prioritized experience replay as I suspected that accumulating lots of low reward experiences would lead to oversampling them and stall learning. However, once I trained the "one-mind" agent for longer episodes I noticed that it got out of its rut after a while. Hence, I decided to postpone the implementation of prioritized replay and try the environment with multiple agents. I started with a naive approach by just training two `DDPG` agents simultaneously on the environment. It turned out that these performed just as well as the "one-mind" agent, solving it after `~7000` episodes.

From looking at poorly performing agents and well performing ones I have the suspicion that various factors contribute to the initial crash of performance. One being that there might be a better way to initialize model parameters - for now I use the default initialization of PyTorch. I'm also suspecting that the reward signal is difficult. When the agents constantly drop the ball they have no indication which of their actions actually got them closer to their desired goal. For instance, one agent might have hit the ball and got it closer to the net but still dropped it, however, the current reward signal would still be just `-0.05` no matter how close it was. This means that agents have to rely on randomly hitting the ball across the net. I'm thinking that prioritized replay might help with that.

While investigating these issues I've extended the command line interface with some convenience functionality. For instance it is now possible to create snapshots once the agent reaches a certain performance level. This is also what I used during training to save the agent model parameters when it actually achieved its highest performance. These are also the model parameters saved as final parameters in this repository. Additionally, I also now properly take care of keyboard interrupts allowing me to stop training in between, save the agent and show the training graph. This helped a lot in investigating issues with training performance. 

## Results
Using a neural network architecture similar to the one used in the [Report_p2_continuous.md](Report_p2_continuous.md), my agent solved the environment after about ~7000 episodes. However, I must say that I do not take the maximum score from both agent rewards, but rather average them. However, this leads to an overall lower score, meaning that the agent might have solved it earlier, but I'm suspecting that the difference is small. 

![Graph of Training Run](../resources/images/nddpg_training.png)

Both agents used the `multi_ddpg_ann_a_2x256_c_2x256_1x128-2020-03-07` agent configuration:
```json
{
  "act_noise_std": 0.1,
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

The final trained model is stored under `resources/models/p3_tennis_final`.

## Future Work
Considering the challenges I faced in this environment, I think the focus for improving performance is more on improving the replayed experience and the exploration/exploitation trade-off. What leads me to this believe, is the fact that the "one-minded" agents exhibits similar performance to the naive multiple-agents approach. One easy improvement could be to find a better way to initialize the model weights, so the agents does better exploring initially. 

Additionally prioritized replay could help the agent to learn more from unusual experiences and reduce the oversampling of what I'd call "states in careful balance". The agent might then be not that "surprised" by unexpected ball trajectories and perform more robustly overall. 

Of course one could also improve the `multi-agent` aspect of it as well and implement the [MADDPG](https://arxiv.org/abs/1706.02275) algorithm (which I intended initially, but turned out not to be necessary). 