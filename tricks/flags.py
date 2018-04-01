import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float("learning_rate", 0.01, 'Initial learning rate.')
flags.DEFINE_integer("epoch_number", None, 'Number of epochs to run trainer.')
flags.DEFINE_integer("batch_size", 1024, "Batch size in a single gpu.")
flags.DEFINE_string("checkpoint_dir", "./checkpoint/", "Indicates the checkpoint directory.")

print "learning_rate is", FLAGS.learning_rate
print "epoch_number", FLAGS.epoch_number
