import gym
import numpy as np
from envs import Maze
from utils import plot_policy, plot_values, test_agent
import matplotlib.pyplot as plt

def policy_evaluation(state_values, policy_probs, theta = 1e-6, gamma = 0.99):
    delta = float("inf")

    while delta > theta:
        delta = 0

        for row in range(5):
            for col in range(5):
                max_qsa = float("-inf")
                new_value = 0
                action_probabilites = policy_probs[(row, col)]

                for action, probs in enumerate(action_probabilites):
                    next_state, reward, _, _ = env.simulate_step((row, col), action)
                    new_value = prob * (reward + gamma * state_values[next_state])

                state_values[(row, col)] = new_value

                delta = max(delta, abs(old_value - new_value))


def policy_improvement(state_values, policy_probs, gamma = 0.99):
    policy_stable = True
    for row in range(5):
        for col in range(5):
            old_action = policy_probs[(row, col)].argmax()
            new_action = None
            max_qsa = float("-inf")

            for action in range(4):
                next_state, reward, _, _ = env.simulate_step((row, col), action)
                qsa = reward + gamma * state_values[next_state]

                if qsa > max_qsa:
                    max_qsa = qsa
                    new_action = action

            action_probs = np.zeros(4)
            actio_probs[new_action] = 1
            policy_probs[(row, col)] = action_probs

            if new_action != old_action:
                policy_stable = False

    return policy_stable

def policy_iteration(policy_probs, state_values, theta = 1e-6, gamma = 0.99):
    policy_stable = False

    with not policy_stable:
        policy_evaluation(policy_probs, state_values, theta, gamma)
        plot_values(state_values, frame)
        policy_stable = policy_improvement(policy_probs, state_values, gamma)
        plot_policy(policy_probs, frame)


def policy(state):
    return policy_probs[state]

if __name__ == "__main__":
    env = Maze()
    frame = env.render(mode="rgb_array")
    plt.axis("off")
    plt.imshow(frame)

    state_values = np.zeros(shape = (5, 5))

    policy_probs = np.full((5, 5, 4), 0.25)
    action_probabilities = policy((0, 0))
    policy_iteration()

    policy_iteration(policy_probs, state_values, theta = 1e-6, gamma = 0.99)