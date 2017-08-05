# neural net for SIFT with tensorflow
# basically stolen from tensorflow tutorial

import tensorflow as tf

x = tf.placeholder(tf.float32, shape=[None, 3072])
y_ = tf.placeholder(tf.float32, shape=[None, 2])


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


# reshape image to tensor (-1 => maintain size)
x_image = tf.reshape(x, [-1, 32, 32, 3])

# make/init conv layer weights; 5x5 px, 1 color channel, 32 filters
W_conv1 = weight_variable([5, 5, 3, 32])
# and init 32 biases(1 per filter_
b_conv1 = bias_variable([32])

# apply rectifier to convolution
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
# apply max_pool; h_pool1.shape=(
h_pool1 = max_pool_2x2(h_conv1)


W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

# apply rectifier to convolution
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
# apply max_pool; h_pool1.shape=(?, 7, 7, 64)
h_pool2 = max_pool_2x2(h_conv2)


# desnsely connected layer
W_dense1 = weight_variable([7 * 7 * 64, 1024])
b_dense1 = bias_variable([1024])


h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_dense1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_dense1) + b_dense1)
# h_dense1 is intermediate result, shape=(?, 1024)

# dropout to reduce overfitting; can be turned on for training and off for test
keep_prob = tf.placeholder(tf.float32)
h_dense1_drop = tf.nn.dropout(h_dense1, keep_prob)

W_dense2 = weight_variable([1024, 10])
b_dense2 = bias_variable([10])

y_conv = tf.matmul(h_dense1_drop, W_dense2) + b_dense2


loss = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
correct_predictions = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(1000):
        batch = mnist.train.next_batch(50)
        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={
                x: batch[0], y_: batch[1], keep_prob: 1.0})
            print('step %d, train_accuracy %g' % (i, train_accuracy))
        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

    print('test accuracy %g' % accuracy.eval(feed_dict={
        x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
