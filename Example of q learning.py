# https://www.youtube.com/watch?v=EnCoYY087Fc&ab_channel=NicolaiNielsen-ComputerVision%26AI

import numpy as np
import pandas as pd
import random

rewards = np.array([
    [-1, -1, -1, -1,  0, -1],
    [-1, -1, -1,  0, -1, 100],
    [-1, -1, -1,  0, -1, -1],
    [-1,  0,  0, -1,  0, -1],
    [ 0, -1, -1,  0, -1, -1],
    [-1,  0, -1, -1, -1, -1]
])

def initialize_q(m, n):
    return np.zeros((m, n))

q_matrix = initialize_q(6, 6)

def set_initial_state(rooms = 6):
    return np.random.randint(0, rooms)

def get_action(current_state, reward_matrix):
    valid_actions = []
    for action in enumerate(reward_matrix[current_state]):
        if action[1] != -1:
            valid_actions += [action[0]]
    
    return random.choice(valid_actions)

def take_action(current_state, reward_matrix, gamma, verbose = False):
    action = get_action(current_state, reward_matrix)
    sa_reward = reward_matrix[current_state, action]
    ns_reward = max(q_matrix[action,])
    q_current_state = sa_reward + (gamma * ns_reward)
    q_matrix[current_state, action] = q_current_state
    new_state = action
    if verbose:
        print(q_matrix)
        print(f"Old State: {current_state} | New State: {new_state}\n\n")
        if new_state == 5:
            print(f"Agent has reached it's goal! 😁")
    return new_state

def initialize_episode(reward_matrix, initial_state, gamma, verbose = False):
    current_state = initial_state
    while True:
        current_state = take_action(current_state, reward_matrix, gamma, verbose)
        if current_state == 5:
            break

def train_agent(iterations, reward_matrix, gamma, verbose = False):
    print("Training in progress...⌛")
    for episode in range(iterations):
        initial_state = set_initial_state()
        initialize_episode(reward_matrix, initial_state, gamma, verbose)
    print("Training complete! 💯")

    return q_matrix

def normalize_matrix(q_matrix):
    normalized_q = q_matrix / max(q_matrix[q_matrix.nonzero()]) * 100
    return normalized_q.astype(int)

'''
gamma = 0.1
initial_state = 2
initial_action = get_action(initial_state, rewards)

initialize_episode(rewards, initial_state, gamma, True)
'''

gamma = 0.8
initial_state = set_initial_state()
initial_action = get_action(initial_state, rewards)

q_table = train_agent(2000, rewards, gamma, False)

print(pd.DataFrame(q_table))
print(pd.DataFrame(normalize_matrix(q_table)))

def deploy_agent(init_state, q_table):
    print('Start: ', init_state)
    state = init_state
    steps = 0
    while True:
        steps += 1
        action = np.argmax(q_table[state,:])
        print(action)
        state = action
        if action == 5:
            print("Finished!")
            return steps

start_room = 0
steps = deploy_agent(start_room, q_table)
print("Number of Rooms/actions: ", steps)