from unityagents import UnityEnvironment
import numpy as np
from dqn_agent import Agent
from collections import deque
import torch
import matplotlib.pyplot as plt

def train(agent, env, brain_name, eps_start=1.0, eps_end=0.005, eps_decay=0.90, max_episodes = 2000, max_steps = 1000, solve_criteria = 13.0):
    """
    Trains a RL agent with unity framework.
    """
    scores = []
    scores_window = deque(maxlen=100)
    eps = eps_start

    for i_episode in range(1, max_episodes + 1):
        
        # This code is slightly different from openai gym
        env_info = env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations[0]
        score = 0.

        for step in range(max_steps):
            
            action = agent.act(state, eps)
            
            # OpenAI code is next_state, reward, done, _ = env.step(action)
            env_info = env.step(action)[brain_name]
            next_state, reward, done = env_info.vector_observations[0], env_info.rewards[0], env_info.local_done[0]

            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break
        
        scores_window.append(score)
        scores.append(score)
        eps = max(eps_end, eps_decay*eps)

        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        if np.mean(scores_window)>= solve_criteria:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')
            break
    
    return scores
            
if __name__ == "__main__":
    env = UnityEnvironment(file_name="envs/Banana_Windows_x86_64/Banana.exe")
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    # reset the environment
    env_info = env.reset(train_mode=True)[brain_name]

    # number of agents in the environment
    print('Number of agents:', len(env_info.agents))

    # number of actions
    action_size = brain.vector_action_space_size
    print('Number of actions:', action_size)

    # examine the state space 
    state = env_info.vector_observations[0]
    print('States look like:', state)
    state_size = len(state)
    print('States have length:', state_size)

    # Creating and training the agent
    agent = Agent(state_size=state_size, action_size=action_size, seed=0)
    scores = train(agent, env, brain_name)
    env.close()

    # plot the scores
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()
               
    