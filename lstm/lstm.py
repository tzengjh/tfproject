# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import os
import sys
scriptpath = "/home/rootuser/githubs/models/tutorials/rnn/ptb"
sys.path.append(os.path.abspath(scriptpath))
import reader
import tensorflow as tf
import numpy as np

# 存放原始数据的路径
DATA_PATH = "/home/rootuser/datasets/ptb/simple-examples/data" # PTB数据保存路径
HIDDEN_SIZE = 200 # 隐含层节点个数
NUM_LAYERS = 2 # LSTM层数
VOCAB_SIZE = 10000

LEARNING_RATE = 1.0
TRAIN_BATCH_SIZE = 20
TRAIN_NUM_STEP = 35

# 测试时只需要将数据视为超长序列
EVAL_BATCH_SIZE = 1
EVAL_NUM_STEP = 1
NUM_EPOCH = 2
KEEP_PROB = 0.5      # Dropout 概率
MAX_GRAD_NORM = 5    # 梯度截断

class PTBModel(object):
    def __init__(self, is_training, batch_size, num_steps):
        self.batch_size = batch_size
        self.num_steps = num_steps

        # inputs
        self.input_data = tf.placeholder(tf.int32, [batch_size, num_steps])
        self.targets = tf.placeholder(tf.int32, [batch_size, num_steps])
        # 定义LSTM结构为循环体结构且使用dropout
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE)
        if is_training:
            lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell,
                                                      output_keep_prob=KEEP_PROB)
        cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * NUM_LAYERS)

        self.initial_state = cell.zero_state(batch_size, dtype=tf.float32)
        embedding = tf.get_variable("embedding", [VOCAB_SIZE, HIDDEN_SIZE])
        # 将 batch_size * num_steps的输入转化为 batch_size * num_steps * HIDDEN_SIZE
        inputs = tf.nn.embedding_lookup(embedding, self.input_data)
        if is_training:
            inputs = tf.nn.dropout(inputs, KEEP_PROB)

        # 将不同时刻LSTM的输出收集,再通过全连接得到最终输出用于求和
        outputs = []
        state = self.initial_state
        with tf.variable_scope("RNN"):
            for time_step in range(num_steps):
                if time_step > 0: tf.get_variable_scope().reuse_variables()
                cell_output, state = cell(inputs[:, time_step, :], state)
                outputs.append(cell_output)

        # 将输出队列展开成 [batch, hidden_size*num_steps]的形状，再reshape为
        # [num_steps*batch_size, hidden_size]的开状
        output = tf.reshape(tf.concat(outputs, 1), [-1, HIDDEN_SIZE])

        weight = tf.get_variable("weight", [HIDDEN_SIZE, VOCAB_SIZE])
        bias = tf.get_variable("bias", [VOCAB_SIZE])
        logits = tf.matmul(output, weight) + bias

        loss = 	tf.contrib.legacy_seq2seq.sequence_loss_by_example(
            [logits],  # 预测结果
            [tf.reshape(self.targets, [-1])], # 目标值
            [tf.ones([batch_size * num_steps], dtype=tf.float32)] # 权重
        )

        self.cost = tf.reduce_sum(loss) / batch_size
        self.final_state = state

        # 只在训练模型时定义反向传播
        if not is_training: return
        trainable_variables = tf.trainable_variables()
        # 通过 clip_by_global_norm 函数控制梯度大小
        grads, _ = tf.clip_by_global_norm(
            tf.gradients(self.cost, trainable_variables), MAX_GRAD_NORM
        )

        optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE)
        self.train_op = optimizer.apply_gradients(
            zip(grads, trainable_variables)
        )

def run_epoch(session, model, data, train_op, output_log):
    total_cost = 0.0
    iters = 0
    state = session.run(model.initial_state)
    print(session.run(reader.ptb_producer(data, model.batch_size, model.num_steps)))

    for step, (x, y) in enumerate(
        reader.ptb_producer(data, model.batch_size, model.num_steps)
        # ptb_iterator
    ):
        cost, state, _ = session.run(
            [model.cost, model.final_state, train_op],
            {model.input_data: x, model.targets: y,
             model.initial_state: state}
        )

        total_cost += cost
        iters += model.num_steps

        if output_log and step % 100 == 0:
            print("After %d steps, perplexity is %.3f" % (
                step, np.exp(total_cost / iters)
            ))
    return  np.exp(total_cost / iters)

def main(_):
    train_data, valid_data, test_data, _ = reader.ptb_raw_data(DATA_PATH)
    initializer = tf.random_uniform_initializer(-0.05, 0.05)

    with tf.variable_scope("language_model", reuse=None, initializer=initializer):
        train_model = PTBModel(True, TRAIN_BATCH_SIZE, TRAIN_NUM_STEP)

    with tf.variable_scope("language_model", reuse=True, initializer=initializer):
        eval_model = PTBModel(False, EVAL_BATCH_SIZE, EVAL_NUM_STEP)

    with tf.Session() as session:
        tf.initialize_all_variables().run()

        for i in range(NUM_EPOCH):
            print("In iteration: %d" % (i + 1))
            run_epoch(session, train_model, train_data, train_model.train_op, True)

            valid_perplexity = run_epoch(session, eval_model, valid_data, tf.no_op(), False)
            print("Epoch: %d Validation Perplexity: %.3f" % (i + 1, valid_perplexity))

        test_perplexity = run_epoch(session, eval_model, test_data, tf.no_op(), False)
        print("Test Perpelxity: %.3f" % test_perplexity)

if __name__ == '__main__':
    tf.app.run()
