import gym
import numpy as np
from utils import plot_policy, plot_values, test_agent
from envs import Maze
import matplotlib.pyplot as plt

def value_iteration(policy_probs, state_values, gamma = 0.99, theta = 1e-6):
    delta = float("inf")

    while delta > theta:
        delta = 0

        for row in range(5):
            for col in range(5):
                old_value = state_values[(row, col)]
                action_probs = None
                max_qsa = float("-inf")
                
                for action in range(4):
                    next_state, reward, _, _ = env.simulate_step((row, col), action)
                    qsa = reward + (gamma * state_values[next_state])
                    
                    if qsa > max_qsa:
                        max_qsa = qsa
                        action_probs = np.zeros(4)
                        action_probs[action] = 1
                    
                state_values[(row, col)] = max_qsa
                policy_probs[(row, col)] = action_probs
                delta = max(delta, abs(max_qsa - old_value))


def policy(state):
    return policy_probs[state]


policy_probs = np.full((5, 5, 4), 0.25)

state_values = np.zeros(shape = (5, 5))

env = Maze()
initial_state = env.reset()
value_iteration(policy_probs, state_values)
test_agent(env, policy, episodes = 1)
env.close()