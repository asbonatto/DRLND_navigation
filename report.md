## Methodology

To complete this task, the most straightforward choice is to use a function approximation to represent the value function as the large number of dimensions make discretization of the state space too costly.  This work implements the [Deep Q Network (DQN)](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf) and the [Double DQN](https://arxiv.org/abs/1509.06461). 

### DQN Recap

The basic Q-learning algorithm iteratively computes the state-action function Q(s,a) using the Bellman update equations:
$ Q(s_{t}, a_{t}) <- (1 - \alpha) * Q(s_{t}, a_{t}) + \alpha * (r_{t+1} + \gamma * max_{a} Q(s_{t+1}, a))$

With Deep Q-Learning, the mapping Q(s,a) is represented by a Deep Neural network. This means we need to train the network while the trainable parameters are used to produce the Q values updated via Bellman equations. This process is unstable, but can be alleviated with two Fixed Q-Targets and Experience Replay. 

With Fixed Q-Targets, the network is not updated every step. The agent uses two "desynchronized" networks : the target and the local network. The target network is kept frozen and is used for updating the Q-values in the Bellman Equation. The local network is then trained with to match the updated values. After a number of "epochs", the networks are synched and the weights of the target network is updated with an average of the local and target parameters.

A second technique used in the original DQN paper is the use of a replay buffer. The technique consists in storing the experience into a memory for posterior sampling. The effectiveness of this method relies on the fact that the state and actions are highly correlated, and the random sampling of the memory breaks this correlation.

While this algorithm is effective, [it was shown that DQN results in overestimation of the Q-values](https://www.ri.cmu.edu/pub_files/pub1/thrun_sebastian_1993_1/thrun_sebastian_1993_1.pdf). This is due to the fact that the Q-values are updated with the max value over noisy values. To avoid this problem, the Double DQN (DDQN) algorithm selects the action with the local network but uses Q-value from the target network.
While this algorithm is effective, [it was shown that DQN results in overestimation of the Q-values](https://www.ri.cmu.edu/pub_files/pub1/thrun_sebastian_1993_1/thrun_sebastian_1993_1.pdf). This is due to the fact that the Q-values are updated with the max value over noisy values. To avoid this problem, the Double DQN (DDQN) algorithm selects the action with the local network but uses Q-value from the target network.

### Implementation and experiments

This repository implements both DQN and DDQN agents. In order to compare the performance of the algorithms, the agents were implemented with the same network architectures and the same hyperparameters. The network consists of two fully connected hidden layers, both using ReLU as activation function, but without activation function in the output layer.

Multiple network architectures were tested : hidden layers of 64 and 32 units. For the training process, the learning rates were kept fixed and equal to 5E-4. Mean square error Huber loss functions were investigated.

Both algorithms uses the epsilon-greedy action selection is implemented with a decay rate of 0.9 per episode and a min eps value of 1E-5, as implemented in the train functions in navigation.py.

### Results

All agents were able to solve task in less than 500 episodes. The convergence characteristics are pretty similar, but one can observe faster convergence in the beginning of the training for the Hubber Loss models with 64 hidden units per layer.

| Model          | # of episodes until convergence|
| -------------- |:------------------------------:|
| hub_ddqn_32_32 | 305                            |
| hub_dqn_64_64  | 280                            |
| msq_ddqn_32_32 | 259                            |
| msq_ddqn_64_64 | 370                            |
| msq_dqn_32_32  | 204                            |
| msq_dqn_64_64  | 300                            |


![hub_ddqn_32_32](experiments\\hub_ddqn_32_32.png)
![hub_dqn_64_64](experiments\\hub_dqn_64_64.png)
![msq_ddqn_32_32](experiments\\msq_ddqn_32_32.png)
![msq_ddqn_64_64](experiments\\msq_ddqn_64_64.png)
![msq_dqn_32_32](experiments\\msq_dqn_32_32.png)
![msq_dqn_64_64](experiments\\msq_dqn_64_64.png)

#### Enhancements

While the proposed architectures were able to solve the problems, a throughout investigation of the hyperparameters such as learning rates and batch sizes is warranted. While this particular task was not too complicated for the standard DQN algorithms, the task could be completed in fewer episodes by DQN enhancements such as [dueling networks](https://arxiv.org/abs/1511.06581) and [prioritized experience replay](https://arxiv.org/abs/1511.05952).