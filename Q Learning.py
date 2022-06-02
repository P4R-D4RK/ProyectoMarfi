import numpy as np
import pandas as pd
import random

table_x = 10
table_y = 10
obstacles = np.array([
    [2,4],
    [0,7],
    [1,5],
    [2,2],
    [6,0],
    [0,4],
    [3,7],
    [5,2],
    [6,5],
    [7,7]
])
final_position = np.array([9, 9])
final_reward = 100
num_random_positions = 50

q_matrix = np.zeros((table_x * table_y * 4, 2))

def exists(value, values):
    for a in values:
        if np.array_equal(a, value):
            return True
    return False

random_positions = []
for _ in range(num_random_positions):
    while True:
        m, n = np.random.randint(table_y), np.random.randint(table_x)
        if not exists([m, n], obstacles) and not exists([m, n], random_positions) and not np.array_equal([m, n], final_position):
            random_positions.append([m, n, np.random.randint(4)])
            break
random_positions = np.array(random_positions)


def get_state_from_position(position):
    return position[0] * 40 + position[1] * 4 + position[2]

# def set_initial_state(random_positions):
#     position = random.choice(random_positions)
#     return position[0] * 40 + position[1] * 4 + np.random.randint(4)

def get_position_from_state(state):
    state = state
    turn = state % 4
    state -= turn
    y = int(state / 40)
    state -= 40 * y
    x = int(state / 4)
    return [y, x, turn]

def get_next_position(current_state):
    y, x, turn = get_position_from_state(current_state)
    if turn == 0:
        return [y - 1, x, turn]
    if turn == 1:
        return [y, x + 1, turn]
    if turn == 2:
        return [y + 1, x, turn]
    if turn == 3:
        return [y, x - 1, turn]

def can_move(current_state, obstacles):
    position = get_next_position(current_state)
    return (position[0] >= 0 and position[0] < table_y) and (position[1] >= 0 and position[1] < table_x) and not exists(position, obstacles)

def get_action(current_state, obstacles):
    return random.randint(0, 1) if can_move(current_state, obstacles) else 1

def take_action(current_state, final_position, obstacles, gamma, verbose = False):
    action = get_action(current_state, obstacles)
    next_position = get_next_position(current_state)
    if action == 1:
        sa_reward = -1
    elif np.array_equal([next_position[0], next_position[1]], final_position):
        sa_reward = 100
    else:
        sa_reward = 0
    ns_reward = max(q_matrix[action,])
    q_current_state = sa_reward + (gamma * ns_reward)
    q_matrix[current_state, action] = q_current_state
    if action == 0: #Avanzar
        new_state = get_state_from_position(get_next_position(current_state))
    else: #Girar
        current_position = get_position_from_state(current_state)
        if current_position[2] == 3:
            current_position[2] = 0
        else:
            current_position[2] += 1
        new_state = get_state_from_position(current_position)
    if verbose:
        print(f"Action {action} | Old State: {current_state} {get_position_from_state(current_state)} | New State: {get_position_from_state(new_state)}")
    return new_state

def initialize_episode(obstacles, initial_state, final_position, gamma, verbose = False):
    current_state = initial_state
    while True:
        current_state = take_action(current_state, final_position, obstacles, gamma, verbose)
        current_position = get_position_from_state(current_state)
        if np.array_equal([current_position[0], current_position[1]], final_position):
            if verbose:
                print('Agent has reached it\'s goal!')
                for i in range(len(q_matrix)):
                    print(f"{i} {q_matrix[i]}")
            break

def train_agent(iterations, obstacles, final_position, gamma, verbose = False):
    print("Training in progress...")
    for i in range(iterations):
        print(f"Training {i + 1} of {iterations}")
        for episode in random_positions:
            initial_state = get_state_from_position(episode)
            initialize_episode(obstacles, initial_state, final_position, gamma, verbose)
    print("Training complete!")  
    return q_matrix

def normalize_matrix(q_matrix):
    normalized_q = q_matrix / max(q_matrix[q_matrix.nonzero()]) * 100
    return normalized_q.astype(int)

q_table = train_agent(50, obstacles, final_position, 0.8, False)
print(q_table)
# print(normalize_matrix(q_table))

def deploy_agent(init_state, final_position, q_table):
    print('Start: ', get_position_from_state(init_state))
    state = init_state
    steps = 0
    while True:
        steps += 1
        action = np.argmax(q_table[state,:])
        
        if action == 0: #Avanzar
            if can_move(state, obstacles):
                state = get_state_from_position(get_next_position(state))
            else:
                current_position = get_position_from_state(state)
                if current_position[2] == 3:
                    current_position[2] = 0
                else:
                    current_position[2] += 1
                state = get_state_from_position(current_position)
        else: #Girar
            current_position = get_position_from_state(state)
            if current_position[2] == 3:
                current_position[2] = 0
            else:
                current_position[2] += 1
            state = get_state_from_position(current_position)
        
        
        current_position = get_position_from_state(state)
        print(current_position)
        if np.array_equal([current_position[0], current_position[1]], final_position):
            print("Finished!")
            return steps

deploy_agent(get_state_from_position(random.choice(random_positions)), final_position, q_table)