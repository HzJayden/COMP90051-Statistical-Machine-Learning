import tensorflow as tf
a = tf.placeholder("float")
b = tf.placeholder("float")
add = tf.add(a, b)
mul = tf.multiply(a, b)
with tf.Session() as sess:
    print(sess.run(add, feed_dict={a: 2, b: 3}))
    print(sess.run(mul, feed_dict=({a: 2, b: 3})))

matrix1 = tf.constant([[3., 3.]])
matrix2 = tf.constant([[2.], [2.]])
product = tf.matmul(matrix1, matrix2)
with tf.Session() as sess:
    result = sess.run(product)
    print(result)
