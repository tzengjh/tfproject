import tensorflow as tf

hello_op = tf.constant("Hello, Tensorflow!")
a = tf.constant(10)
b = tf.constant(32)
compute_op = tf.add(a,b)


with tf.Session() as sess:
    print(sess.run(hello_op))
    print(sess.run(compute_op))

