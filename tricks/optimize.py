import tensorflow as tf

LEARNING_RATE = 0.01
flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("optimizer", "sgc", "Optimize method.")

if FLAGS.optimizer == "sgd":
    optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE)
elif FLAGS.optimizer == "momentum":
    optimizer = tf.train.MomentumOptimizer(LEARNING_RATE)
elif FLAGS.optimizer == "adadelta":
    optimizer = tf.train.AdadeltaOptimizer(LEARNING_RATE)
elif FLAGS.optimizer == "adagrad":
    optimizer = tf.train.AdagradDAOptimizer(LEARNING_RATE)
elif FLAGS.optimizer == "adam":
    optimizer = tf.train.AdamOptimizer(LEARNING_RATE)
elif FLAGS.optimizer == "ftrl":
    optimizer = tf.train.FtrlOptimizer(LEARNING_RATE)
elif FLAGS.optimizer == "rmsprop":
    optimizer = tf.train.RMSPropOptimizer(LEARNING_RATE)
else:
    print("Unknown optimizer:{}, exit now.".format(FLAGS.optimizer))
    exit(1)
