from __future__ import absolute_import, division, print_function

import tensorflow as tf

# enable the eager execution
tf.enable_eager_execution(())

tf.executing_eagerly()

x = [[2]]
m = tf.matmul(x,x)
print("hello {}".format(m))

