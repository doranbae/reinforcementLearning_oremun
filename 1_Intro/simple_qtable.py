import numpy as np
import random


# Define parameters
gamma        = 0.8
num_episode  = 1000
finish_state = 24

# Define reward table
reward = np.array(
    [[ 0,  0,  0,  0,  0 ],
     [ 0,  0,  0,-10,  0 ],
     [ 0,  0,  0,  0,-10 ],
     [ 0,  0,-10,  0,  0 ],
     [ 0,  0,  0,  0,  0 ],
     [ 0,  0,  0,  0,  0 ],
     [ 0,  0,  0,-10,  0 ],
     [ 0,  0,  0,  0,-10 ],
     [ 0,  0,-10,  0,  0 ],
     [ 0,-10,  0,  0,  0 ],
     [ 0,  0,  0,  0,  0 ],
     [ 0,  0,  0,  0,  0 ],
     [ -10,-10,0,  0,  0 ],
     [ 0,  0,  0,-10,  0 ],
     [ 0,  0,  0,  0,-10 ],
     [ 0,-10,  0,  0,  0 ],
     [ 0,  0,  0,-10,  0 ],
     [ 0,  0,  0,  0,-10 ],
     [ 0,  0,-10,  0,  0 ],
     [-10,10,  0,  0,  0 ],
     [ 0,  0,  0,  0,-10 ],
     [ 0,  0,-10,  0,  0 ],
     [-10, 0,  0,  0,  0 ],
     [ 0,  0,  0, 10,  0 ],
     [ 0,  0,  0,  0, 10 ]]
)

# Define transition matrix
transition_matrix = np.array(
    [[ -1, 5 , -1,  1,  0 ],
     [ -1, 6 ,  0,  2,  1 ],
     [ -1, 7 ,  1,  3,  2 ],
     [ -1, 8 ,  2,  4,  3 ],
     [ -1, 9 ,  3, -1,  4 ],
     [ 0 , 10, -1,  6,  5 ],
     [ 1 , 11,  5,  7,  6 ],
     [ 2 , 12,  6,  8,  7 ],
     [ 3 , 13,  7,  9,  8 ],
     [ 4 , 14,  8, -1,  9 ],
     [ 5 , 15, -1, 11, 10 ],
     [ 6 , 16, 10, 12, 11 ],
     [ 7 , 17, 11, 13, 12 ],
     [ 8 , 18, 12, 14, 13 ],
     [ 9 , 19, 13, -1, 14 ],
     [ 10, 20, -1, 16, 15 ],
     [ 11, 21, 15, 17, 16 ],
     [ 12, 22, 16, 18, 17 ],
     [ 13, 23, 17, 19, 18 ],
     [ 14, 24, 18, -1, 19 ],
     [ 15, -1, -1, 21, 20 ],
     [ 16, -1, 20, 22, 21 ],
     [ 17, -1, 21, 23, 22 ],
     [ 18, -1, 22, 24, 23 ],
     [ 19, -1, 23, -1, 24 ]]
)


# Define valid actions
# Encoded up = 0, down = 1, left = 2, right = 3, no action = 4
valid_actions = np.array(
    [[ 1, 3, 4 ],
     [ 1, 2, 3, 4 ],
     [ 1, 2, 3, 4 ],
     [ 1, 2, 3, 4 ],
     [ 1, 2, 4 ],
     [ 0, 1, 3, 4 ],
     [ 0, 1, 2, 3, 4 ],
     [ 0, 1, 2, 3, 4 ],
     [ 0, 1, 2, 3, 4 ],
     [ 0, 1, 2, 4 ],
     [ 0, 1, 3, 4 ],
     [ 0, 1, 2, 3, 4 ],
     [ 0, 1, 2, 3, 4 ],
     [ 0, 1, 2, 3, 4 ],
     [ 0, 1, 2, 4 ],
     [ 0, 1, 3, 4 ],
     [ 0, 1, 2, 3, 4 ],
     [ 0, 1, 2, 3, 4 ],
     [ 0, 1, 2, 3, 4 ],
     [ 0, 1, 2, 4 ],
     [ 0, 3, 4 ],
     [ 0, 2, 3, 4 ],
     [ 0, 2, 3, 4 ],
     [ 0, 2, 3, 4 ],
     [ 0, 2, 4 ]]
)


# Initialize q_matrix with 0s.
q_matrix = np.zeros((25,5))

for i in range(num_episode):
    start_state = 0
    curr_state  = start_state
    while curr_state  != finish_state:                                         # While it is not yet finished
        action         = random.choice(valid_actions[curr_state])              # Choose a random action from possible U/D/R/L/N
        next_state     = transition_matrix[curr_state][action]                 # Look up what the next tile is (after I take the action)
        future_rewards = []
        for action_nxt in valid_actions[next_state]:                           # From all valid actions from the next move
            future_rewards.append(q_matrix[next_state][action_nxt])            # Get the future rewards
        q_state = reward[curr_state][action] + gamma * max(future_rewards)
        q_matrix[curr_state][action] = q_state                                 # Update the reward
        curr_state = next_state                                                # Move on

print('Final q-matrix:')
print(q_matrix)