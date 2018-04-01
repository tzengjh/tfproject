import tensorflow as tf

checkpoint_dir = "./"

with tf.Session() as sess:
    if mode == "train" or mode == "train_from_scratch":
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
