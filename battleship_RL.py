import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 1.2 Define the nn variable network.
# Input is array of BOARD_SIZE values.
# ---------------------------------------
#  -1 value -> Not yet checked
#   0 value -> Checked, no ship
#   1 value -> Checked, is ship location.
# ---------------------------------------
BOARD_SIZE = 10
SHIP_SIZE = 3

hidden_units = BOARD_SIZE
output_units = BOARD_SIZE

# positions of the game board
input_positions = tf.placeholder(tf.float32, shape=(1, BOARD_SIZE))

labels = tf.placeholder(tf.int64)
learning_rate = tf.placeholder(tf.float32, shape=[])


def add_layer(inputs, in_size, out_size, activation_function=None):
    weights = tf.Variable(tf.truncated_normal([in_size, out_size],
                                              stddev=0.1 / np.sqrt(float(in_size))))
    biases = tf.Variable(tf.zeros([1, out_size]))
    wx_plus_b = tf.matmul(inputs, weights) + biases
    if activation_function is None:
        return wx_plus_b
    else:
        return activation_function(wx_plus_b)


# Generate hidden layer
h1 = add_layer(input_positions, BOARD_SIZE, hidden_units, activation_function=tf.tanh)
# Second layer -- linear classifier for action logits
h2 = add_layer(h1, hidden_units, output_units)
probabilities = tf.nn.softmax(h2)

# 1.3 Define the operations we will use
init = tf.global_variables_initializer()
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
    logits=h2, labels=labels, name='xentropy')  # loss
train_step = tf.train.GradientDescentOptimizer(
    learning_rate=learning_rate).minimize(cross_entropy)
# Start TF session
sess = tf.Session()
sess.run(init)

# 1.4 Game play definition.
TRAINING = True


def play_game(training=TRAINING):
    """ Play game of battleship using network."""
    # Select random location for ship
    ship_left = np.random.randint(BOARD_SIZE - SHIP_SIZE + 1)
    ship_positions = set(range(ship_left, ship_left + SHIP_SIZE))
    # Initialize logs for game
    board_position_log = []
    action_log = []
    hit_log = []
    # Play through game
    current_board = [[-1 for i in range(BOARD_SIZE)]]
    while (sum(hit_log) < SHIP_SIZE) and (len(action_log) < BOARD_SIZE):
        board_position_log.append([[i for i in current_board[0]]])
        probs = sess.run([probabilities], feed_dict={input_positions: current_board})[0][0]
        probs = [p * (index not in action_log) for index, p in enumerate(probs)]
        probs = [p / sum(probs) for p in probs]
        if training:
            bomb_index = np.random.choice(BOARD_SIZE, p=probs)
        else:
            bomb_index = np.argmax(probs)
        # update board, logs
        hit_log.append(1 * (bomb_index in ship_positions))
        current_board[0][bomb_index] = 1 * (bomb_index in ship_positions)
        action_log.append(bomb_index)
    # print(board_position_log, action_log, hit_log)
    return board_position_log, action_log, hit_log


# 1.5 Reward function definition
def rewards_calculator(hit_log, gamma=0.5):
    """ Discounted sum of future hits over trajectory"""
    hit_log_weighted = [(item -
                         float(SHIP_SIZE - sum(hit_log[:index])) / float(BOARD_SIZE - index)) * (
                            gamma ** index) for index, item in enumerate(hit_log)]
    return [((gamma) ** (-i)) * sum(hit_log_weighted[i:]) for i in range(len(hit_log))]


# 1.6 Training loop: Play and learn
game_lengths = []
TRAINING = True  # Boolean specifies training mode
ALPHA = 0.06  # step size

for game in range(10000):
    board_position_log, action_log, hit_log = play_game(training=TRAINING)
    game_lengths.append(len(action_log))
    rewards_log = rewards_calculator(hit_log)
    for reward, current_board, action in zip(rewards_log, board_position_log, action_log):
        # Take step along gradient
        if TRAINING:
            sess.run([train_step],
                     feed_dict={input_positions: current_board, labels: [action], learning_rate: ALPHA * reward})

# 1.7 Plot running average game lengths
window_size = 500
running_average_length = [np.mean(game_lengths[i:i+window_size]) for i in range(len(game_lengths)- window_size)]
plt.plot(running_average_length)
plt.show()
