[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"

# Project 1: Navigation

### Introduction

The goal of this project is to develop an autonomous agent capable of collecting yellow bananas while avoiding the blue ones.

![Trained Agent][image1]

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.  Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.  

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions.  Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The task is episodic, and in order to solve the environment, your agent must get an average score of +13 over 100 consecutive episodes.

### Methodology

To complete this task, the most straightforward choice is to use a function approximation to represent the value function as the large number of dimensions make discretization of the state space too costly.  This work implements the [Deep Q Network (DQN)](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf) and the [Double DQN](https://arxiv.org/abs/1509.06461). 

#### DQN Recap

The basic Q-learning algorithm iteratively computes the state-action function Q(s,a) using the Bellman update equations:
$ Q(s_{t}, a_{t}) <- (1 - \alpha) * Q(s_{t}, a_{t}) + \alpha * (r_{t+1} + \gamma * max_{a} Q(s_{t+1}, a))$

With Deep Q-Learning, the mapping Q(s,a) is represented by a Deep Neural network. This means we need to train the network while the trainable parameters are used to produce the Q values updated via Bellman equations. This process is unstable, but can be alleviated with two Fixed Q-Targets and Experience Replay. 

With Fixed Q-Targets, the network is not updated every step. The agent uses two "desynchronized" networks : the target and the local network. The target network is kept frozen and is used for updating the Q-values in the Bellman Equation. The local network is then trained with to match the updated values. After a number of "epochs", the networks are synched and the weights of the target network is updated with an average of the local and target parameters.

A second technique used in the original DQN paper is the use of a replay buffer. The technique consists in storing the experience into a memory for posterior sampling. The effectiveness of this method relies on the fact that the state and actions are highly correlated, and the random sampling of the memory breaks this correlation.

While this algorithm is effective, [it was shown that DQN results in overestimation of the Q-values](https://www.ri.cmu.edu/pub_files/pub1/thrun_sebastian_1993_1/thrun_sebastian_1993_1.pdf). This is due to the fact that the Q-values are updated with the max value over noisy values. To avoid this problem, the Double DQN (DDQN) algorithm selects the action with the local network but uses Q-value from the target network.

#### Implementation and experiments

This repository implements both DQN and DDQN agents. In order to compare the performance of the algorithms, the agents were implemented with the same network architectures and the same hyperparameters. The network consists of two fully connected hidden layers, both using ReLU as activation function, but without activation function in the output layer.

Multiple network architectures were tested : hidden layers of 64 and 32 units. For the training process, the learning rates were kept fixed and equal to 5E-4. Mean square error Huber loss functions were investigated.

Both algorithms uses the epsilon-greedy action selection is implemented with a decay rate of 0.9 per episode and a min eps value of 1E-5, as implemented in the train functions in navigation.py.

#### Results

All agents were able to solve task in less than 500 episodes. The convergence characteristics are pretty similar, but one can observe faster convergence in the beginning of the training for the Hubber Loss models with 64 hidden units per layer.

![hub_ddqn_32_32](experiments\\hub_ddqn_32_32.png)
![hub_dqn_64_64](experiments\\hub_dqn_64_64.png)
![msq_ddqn_32_32](experiments\\msq_ddqn_32_32.png)
![msq_ddqn_64_64](experiments\\msq_ddqn_64_64.png)
![msq_dqn_32_32](experiments\\msq_dqn_32_32.png)
![msq_dqn_64_64](experiments\\msq_dqn_64_64.png)

#### Enhancements

While the proposed architectures were able to solve the problems, a throughout investigation of the hyperparameters such as learning rates and batch sizes is warranted. While this particular task was not too complicated for the standard DQN algorithms, the task could be completed in fewer episodes by DQN enhancements such as [dueling networks](https://arxiv.org/abs/1511.06581) and [prioritized experience replay](https://arxiv.org/abs/1511.05952).


### Installing and Running the Environment 

1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip) to obtain the environment.

2. Place the file in the DRLND GitHub repository, in the `p1_navigation/` folder, and unzip (or decompress) the file. 

### Instructions

To train the agent, run

```bash
python navigation.py
```

### (Optional) Challenge: Learning from Pixels

After you have successfully completed the project, if you're looking for an additional challenge, you have come to the right place!  In the project, your agent learned from information such as its velocity, along with ray-based perception of objects around its forward direction.  A more challenging task would be to learn directly from pixels!

To solve this harder task, you'll need to download a new Unity environment.  This environment is almost identical to the project environment, where the only difference is that the state is an 84 x 84 RGB image, corresponding to the agent's first-person view.  (**Note**: Udacity students should not submit a project with this new environment.)

You need only select the environment that matches your operating system:
- Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/VisualBanana_Linux.zip)
- Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/VisualBanana.app.zip)
- Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/VisualBanana_Windows_x86.zip)
- Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/VisualBanana_Windows_x86_64.zip)

Then, place the file in the `p1_navigation/` folder in the DRLND GitHub repository, and unzip (or decompress) the file.  Next, open `Navigation_Pixels.ipynb` and follow the instructions to learn how to use the Python API to control the agent.

(_For AWS_) If you'd like to train the agent on AWS, you must follow the instructions to [set up X Server](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the environment for the **Linux** operating system above.
