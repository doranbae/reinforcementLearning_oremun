import numpy as np
import random
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import InputLayer

# Define parameters
alpha        = 0.95
epsilon      = 0.5
gamma        = 0.999
q_avg_list   = []
num_episodes = 100
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


def qtable_nn_model():
    # build model
    model = Sequential()
    model.add(InputLayer(batch_input_shape = (1, 25)))
    model.add(Dense( 50, activation='sigmoid' ))
    model.add(Dense( 5, activation='linear' ))

    # compile model
    model.compile(loss = 'mse', optimizer='adam', metrics=['mae'])
    return model

model = qtable_nn_model()

# execute the q learning
for i in range(num_episodes):
    curr_state = 0
    epsilon   *= gamma
    if i % 10 == 0:
        print('Episode {} of {}'.format(i + 1, num_episodes))
    r_sum = 0
    while curr_state != finish_state:
        if random.uniform(0,1) < epsilon:
            action = random.choice(valid_actions[curr_state])
        else:
            possible_action_candidates = model.predict( np.identity(25)[ curr_state : curr_state + 1 ])
            valid_action_candidates = [x if transition_matrix[curr_state][idx] != -1 else -10 for idx, x in enumerate(possible_action_candidates[0])]
            action = np.argmax(valid_action_candidates)

        next_state = transition_matrix[curr_state][action]
        r          = reward[curr_state][action]

        target = r + alpha * np.max( model.predict(np.identity(25)[next_state : next_state + 1 ]) )
        target_vec = model.predict( np.identity(25)[curr_state: curr_state + 1]  )[0]
        target_vec[action] = target

        model.fit(np.identity(25)[curr_state: curr_state + 1] , target_vec.reshape(-1, 5 ), epochs = 10, verbose = 0)
        curr_state = next_state
        r_sum += r
    q_avg_list.append(r_sum / 1000)

print(q_avg_list)

# Evaluation
print('Evaluation')
print(' ')
print('state 2: right is trap')
print(model.predict(np.identity(25)[1:2]))
print('-------------------------------------------------')
print('state 23: right is finish')
print(model.predict(np.identity(25)[23:24]))
print('-------------------------------------------------')
print('state 13: up and bottom is trap')
print(model.predict(np.identity(25)[12:13]))
print('-------------------------------------------------')
print('state 20: up is trap and bottom is finish')
print(model.predict(np.identity(25)[19:20]))


# Find path
start = 0
finish = 24
curr_path = start
best_path = [start]

while curr_path != finish:
    best_action = np.argmax(model.predict(np.identity(25)[curr_path: curr_path + 1]))
    next_path   = transition_matrix[curr_path][best_action]
    best_path.append(next_path)
    curr_path = next_path


print(best_path)