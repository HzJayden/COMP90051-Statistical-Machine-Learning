import numpy as np
import tensorflow as tf
import csv
from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.utils import shuffle

data = np.load('data.npz')
train_X_scaled = preprocessing.scale(data['train_X'])
train_y = data['train_y'].astype(int)
test_X_scaled = preprocessing.scale(data['test_X'])

onehot_labels = np.zeros((train_y.shape[0], 10))
onehot_labels[np.arange(train_y.shape[0]), train_y] = 1

with tf.name_scope('input') as scope:
    test = tf.placeholder(tf.float32, shape=[None, 4096], name='test')
    x = tf.placeholder(tf.float32, shape=[None, 4096], name='train_x')
    y_ = tf.placeholder(tf.float32, shape=[None, 10], name='train_y')


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


# image width and height, number of color channels
with tf.name_scope('reshape'):
    x_image = tf.reshape(x, [-1, 64, 64, 1],name='reshape')

with tf.name_scope('conv1'):
    # patch size, input channels, output channels.
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1, name='conv1')

with tf.name_scope('pool1'):
    h_pool1 = max_pool_2x2(h_conv1)

with tf.name_scope('conv2'):
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2, name='conv2')

with tf.name_scope('pool2'):
    h_pool2 = max_pool_2x2(h_conv2)

with tf.name_scope('conv3'):
    W_conv3 = weight_variable([5, 5, 64, 128])
    b_conv3 = bias_variable([128])
    h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)

with tf.name_scope('pool3'):
    h_pool3 = max_pool_2x2(h_conv3)

with tf.name_scope('conv4'):
    W_conv4 = weight_variable([5, 5, 128, 256])
    b_conv4 = bias_variable([256])
    h_conv4 = tf.nn.relu(conv2d(h_pool3, W_conv4) + b_conv4)

with tf.name_scope('pool4'):
    h_pool4 = max_pool_2x2(h_conv4)

with tf.name_scope('fc1'):
    W_fc1 = weight_variable([4 * 4 * 256, 1024])
    b_fc1 = bias_variable([1024])
    h_pool4_flat = tf.reshape(h_pool4, [-1, 4 * 4 * 256])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool4_flat, W_fc1) + b_fc1)

with tf.name_scope('dropout'):
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

with tf.name_scope('fc2'):
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])
    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

predict = tf.arg_max(y_conv, 1)
with tf.name_scope('loss'):
    cross_entropy = \
        tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))

with tf.name_scope('adam_optimizer'):
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

epochs = 50
batch_size = 100

losses = []
# Add ops to save and restore all the variables.
saver = tf.train.Saver()

with tf.Session() as sess:
    writer = tf.summary.FileWriter('./graphs', sess.graph)

    sess.run(tf.global_variables_initializer())

    for epoch in range(epochs):
        i = 0
        # shuffle training data set
        train_X_scaled, onehot_labels = shuffle(train_X_scaled, onehot_labels, random_state=0)
        while i < train_X_scaled.shape[0]:
            start = i
            end = i + batch_size
            batch_X = train_X_scaled[start:end, :]
            batch_y = onehot_labels[start:end, :]
            train_step.run(feed_dict={x: batch_X, y_: batch_y, keep_prob: 0.5})
            i = end
            Loss = cross_entropy.eval(feed_dict={x: batch_X, y_: batch_y, keep_prob: 1.0})
            print('Loss: ', Loss)
            losses.append(Loss)

        print("epoch:", epoch)

    classification = sess.run(predict, feed_dict={x:test_X_scaled, keep_prob:1.0})
    print(classification)

    headers = ['Id', 'Label']
    rows = [(id + 1, label) for id, label in enumerate(classification)]

    with open('./prediction.csv', 'w', newline='') as f:
        f_csv = csv.writer(f)
        f_csv.writerow(headers)
        f_csv.writerows(rows)

    save_path = saver.save(sess, "cnn_net/model.ckpt")
    print("Save to path: ", save_path)

# plot the graph
plt.plot(losses, label='Batch Loss')
plt.legend()
plt.xlabel('Batches')
plt.ylabel('Loss')
plt.show()

writer.close()