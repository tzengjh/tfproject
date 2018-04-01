import tensorflow as tf

FEATURE_SIZE = 128
LABEL_SIZE = 2

input_units = FEATURE_SIZE
hidden1_units = 10
hidden2_units = 10
hidden3_units = 10
hidden4_units = 10
output_units = LABEL_SIZE

def full_connect(inputs, weights_shape, biases_shape):
    with tf.device('/cpu:0'):
        weights = tf.get_variable("weights", weights_shape, initializer=tf.random_normal_initializer())
        biases = tf.get_variable("biases", biases_shape, initializer=tf.random_normal_intializer())
        return tf.matmul(inputs, weights) + biases

def full_connect_relu(inputs, weights_shape, biases_shape):
    return tf.nn.relu(full_connect(inputs, weights_shape, biases_shape))

def deep_inference(inputs):
    with tf.variable_scope("layer1"):
        layer = full_connect_relu(inputs, [input_units, hidden1_units], [hidden1_units])

    with tf.variable_scope("layer2"):
        layer = full_connect_relu(layer, [hidden1_units, hidden2_units], [hidden2_units])

    with tf.variable_scope("layer3"):
        layer = full_connect_relu(layer, [hidden2_units, hidden3_units], [hidden3_units])

    with tf.variable_scope("layer4"):
        layer = full_connect_relu(layer, [hidden3_units, hidden4_units], [hidden4_units])

    with tf.variable_scope("output"):
        layer = full_connect_relu(layer, [hidden4_units, output_units], [output_units])

    return layer 
