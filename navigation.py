from unityagents import UnityEnvironment
import numpy as np
from dqn_agent import Agent
from collections import deque
import torch
import pandas as pd
import os

CHECKPOINTS_DIR = "experiments"

class EnvWrapper():
    """
    Wrapper for unity framework to match OpenAI environment interface
    """
    def __init__(self, env):
        self.env = env
        self.brain_name = env.brain_names[0]
        
        brain = env.brains[self.brain_name]
        
        env_info = self.env.reset(train_mode=True)[self.brain_name]

        self.nA = brain.vector_action_space_size
        print('Number of actions:', self.nA)
        
        # number of agents in the environment
        print('Number of agents:', len(env_info.agents))
        # examine the state space 
        state = env_info.vector_observations[0]
        print('States look like:', state)
        self.nS = len(state)
        print('State space dimension:', self.nS)

    def reset(self):
        """
        Returns the state, as OpenAI env would
        """
        env_info = self.env.reset(train_mode = True)[self.brain_name]
        return env_info.vector_observations[0]
    
    def step(self, action):
        """
        Updates the environment with action and sends feedback to the agent
        """
        env_info = self.env.step(action)[self.brain_name]
        next_state, reward, done = env_info.vector_observations[0], env_info.rewards[0], env_info.local_done[0]
        return next_state, reward, done

    def close(self):
        self.env.close()


def train(agent, env, eps_start=1.0, eps_end=1E-5, eps_decay=0.90, max_episodes = 1000, max_steps = 500, solve_criteria = 13.0):
    """
    Trains a RL agent with unity framework.
    """
    scores = []
    scores_window = deque(maxlen=100)
    eps = eps_start

    for i_episode in range(1, max_episodes + 1):
        
        state = env.reset()
        score = 0.

        for step in range(max_steps):
            
            action = agent.act(state, eps)
            next_state, reward, done = env.step(action)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break
        
        scores_window.append(score)
        scores.append([score, step + 1])
        eps = max(eps_end, eps_decay*eps)

        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        if np.mean(scores_window)>= solve_criteria:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
            filename = os.path.join(CHECKPOINTS_DIR, agent.name + '.pth')
            torch.save(agent.qnetwork_local.state_dict(), filename)
            break
    
    return pd.DataFrame(scores, columns = ["score", "episodes"])

if __name__ == "__main__":
    
    env = EnvWrapper(UnityEnvironment(file_name="envs/Banana_Windows_x86_64/Banana.exe"))
    for is_ddqn in [False]:
        # Creating and training the agent
        
        agent = Agent(state_size=env.nS, action_size=env.nA, seed=0, is_ddqn=is_ddqn)
        scores = train(agent, env)
        

        # plot the scores
        filename = os.path.join(CHECKPOINTS_DIR, agent.name + ".{}")

        scores["avg_score"] = scores["score"].rolling(20).mean()
        scores.to_csv(filename.format("csv"))
        cfg = {
            "xlabel" : 'Episode #',
            "ylabel": 'Score'
            }
        ax = scores[["score", "avg_score"]].plot(title = agent.name)
        ax.set(**cfg)
        ax.get_figure().savefig(filename.format("png"))
    env.close()
               
    