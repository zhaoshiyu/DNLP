# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn

class RNNModel():
    def __init__(self, args):
        self.args = args

        if self.args.model == 'rnn':
            cell_fn = rnn.BasicRNNCell
        elif self.args.model == 'gru':
            cell_fn = rnn.GRUCell
        elif self.args.model == 'lstm':
            cell_fn = rnn.BasicLSTMCell
        else:
            raise Exception('model type not supported: {}'.format(self.args.model))

        cell = cell_fn(self.args.rnn_size, self.args.state_is_tuple)
        cell = rnn.MultiRNNCell([cell] * self.args.num_layers, self.args.state_is_tuple)

        if self.args.keep_prob < 1:
            cell = rnn.DropoutWrapper(cell, output_keep_prob=self.args.keep_prob)
        self.cell = cell

        self.input_data = tf.placeholder(tf.int64, [None, self.args.seq_length], name='input_data')
        self.targets = tf.placeholder(tf.int64, [None, ], name='targets')  # target is class label
        self.initial_state = cell.zero_state(self.args.batch_size, tf.float32)

        with tf.variable_scope('embeddingLayer'):
            with tf.device('/cpu:0'):
                W = tf.get_variable('W', [self.args.vocab_size, self.args.rnn_size])
                embedded = tf.nn.embedding_lookup(W, self.input_data)
                inputs = tf.split(embedded, self.args.seq_length, 1)
                inputs = [tf.squeeze(input_, [1]) for input_ in inputs]
        
        outputs, last_state = rnn.static_rnn(cell, inputs, self.initial_state, scope='rnnLayer')

        with tf.variable_scope('softmaxLayer'):
            softmax_w = tf.get_variable('w', [self.args.rnn_size, self.args.label_size])
            softmax_b = tf.get_variable('b', [self.args.label_size])
            logits = tf.matmul(outputs[-1], softmax_w) + softmax_b
            self.probs = tf.nn.softmax(logits)
        
        with tf.variable_scope("loss"):
            self.loss = tf.reduce_mean(tf.contrib.nn.deprecated_flipped_sparse_softmax_cross_entropy_with_logits(logits, self.targets), name='loss')  # Softmax loss
        self.final_state = last_state
        self.lr = tf.Variable(0.0, trainable=False)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)  # Adam Optimizer
        #self.optimizer = tf.train.AdamOptimizer(learning_rate=self.args.learning_rate).minimize(self.cost)  # Adam Optimizer

        self.correct_pred = tf.equal(tf.argmax(self.probs, 1), self.targets)
        self.correct_num = tf.reduce_sum(tf.cast(self.correct_pred, tf.float32))
        with tf.variable_scope("accuracy"):
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32), name='accuracy')


    def _predict(self, sess, x):
        state = sess.run(self.cell.zero_state(len(x), tf.float32))
        feed = {self.input_data: x, self.initial_state: state}
        probs, state = sess.run([self.probs, self.final_state], feed_dict=feed)
        return np.argmax(probs, 1)


    def predict_label(self, sess, id2labels, text):
        x = np.array(text)
        x_len = len(x)
        if x_len == self.args.batch_size:
            return list(map(id2labels.get, self._predict(sess, x)))
        if x_len < self.args.batch_size:
            x = np.concatenate((x, np.array([x[-1]] * (self.args.batch_size - x_len))))
            return list(map(id2labels.get, self._predict(sess, x)[:len(text)]))
        else:
            n_chunks = x_len / self.args.batch_size
            if x_len % self.args.batch_size:
                n_chunks += 1
            data_list = np.array_split(x[:self.args.batch_size*n_chunks], n_chunks, axis=0)
            results = []
            for m in range(n_chunks):
                results.extend(self.predict_label(sess, id2labels, data_list[m]))
            return results

    def predict_class(self, sess, text):
        x = np.array(text)
        if len(x) != self.args.batch_size:
            x =np.concatenate((x, np.array([x[-1]] * (self.args.batch_size - len(x)))))
            return self._predict(sess, x)[:len(text)]
        else:
            return self._predict(sess, x)


class BIDIRNNModel():
    def __init__(self, args):
        self.args = args

        if self.args.model == 'rnn':
            cell_fn = rnn.BasicRNNCell
        elif self.args.model == 'gru':
            cell_fn = rnn.GRUCell
        elif self.args.model == 'lstm':
            cell_fn = rnn.BasicLSTMCell
        else:
            raise Exception('model type not supported: {}'.format(self.args.model))
        
        fw_cell = cell_fn(self.args.rnn_size, self.args.state_is_tuple)
        fw_cell = rnn.MultiRNNCell([fw_cell] * self.args.num_layers, self.args.state_is_tuple)
        bw_cell = cell_fn(self.args.rnn_size, self.args.state_is_tuple)
        bw_cell = rnn.MultiRNNCell([bw_cell] * self.args.num_layers, self.args.state_is_tuple)

        if args.keep_prob < 1:
            fw_cell = rnn.DropoutWrapper(fw_cell, output_keep_prob=self.args.keep_prob)
            bw_cell = rnn.DropoutWrapper(bw_cell, output_keep_prob=self.args.keep_prob)
        self.fw_cell = fw_cell
        self.bw_cell = bw_cell

        self.input_data = tf.placeholder(tf.int64, [None, self.args.seq_length], name='input_data')
        self.targets = tf.placeholder(tf.int64, [None, ], name='targets')  # target is class label
        self.initial_state_fw = fw_cell.zero_state(self.args.batch_size, tf.float32)
        self.initial_state_bw = bw_cell.zero_state(self.args.batch_size, tf.float32)
        
        with tf.variable_scope('embeddingLayer'):
            with tf.device('/cpu:0'):
                W = tf.get_variable('W', [self.args.vocab_size, self.args.rnn_size])
                embedded = tf.nn.embedding_lookup(W, self.input_data)
                inputs = tf.split(embedded, self.args.seq_length, 1)
                inputs = [tf.squeeze(input_, [1]) for input_ in inputs]
        
        used = tf.sign(tf.reduce_max(tf.abs(inputs), reduction_indices=2))
        self.length = tf.cast(tf.reduce_sum(used, reduction_indices=0), tf.int32)
        outputs, last_state_fw, last_state_bw = rnn.static_bidirectional_rnn(fw_cell, bw_cell,
                                               inputs, initial_state_fw = self.initial_state_fw,
                                               initial_state_bw = self.initial_state_bw,
                                               dtype=tf.float32, sequence_length=self.length)

        with tf.variable_scope('softmaxLayer'):
            softmax_w = tf.get_variable('w', [self.args.rnn_size*2, self.args.label_size])
            softmax_b = tf.get_variable('b', [self.args.label_size])
            logits = tf.matmul(outputs[-1], softmax_w) + softmax_b
            self.probs = tf.nn.softmax(logits)
        
        with tf.variable_scope("loss"):
            self.loss = tf.reduce_mean(tf.contrib.nn.deprecated_flipped_sparse_softmax_cross_entropy_with_logits(logits, self.targets), name="loss")  # Softmax loss
        self.final_state_fw = last_state_fw
        self.final_state_bw = last_state_bw
        self.lr = tf.Variable(0.0, trainable=False)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)  # Adam Optimizer
        #self.optimizer = tf.train.AdamOptimizer(learning_rate=self.args.learning_rate).minimize(self.cost)  # Adam Optimizer

        self.correct_pred = tf.equal(tf.argmax(self.probs, 1), self.targets)
        self.correct_num = tf.reduce_sum(tf.cast(self.correct_pred, tf.float32))
        with tf.variable_scope("accuracy"):
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32), name="accuracy")


    def _predict(self, sess, x):
        state_fw = sess.run(self.fw_cell.zero_state(len(x), tf.float32))
        state_bw = sess.run(self.bw_cell.zero_state(len(x), tf.float32))
        feed = {self.input_data: x, self.initial_state_fw: state_fw, self.initial_state_bw: state_bw}
        probs, state_fw, state_bw = sess.run([self.probs, self.final_state_fw, self.final_state_bw], feed_dict=feed)
        return np.argmax(probs, 1)


    def predict_label(self, sess, id2labels, text):
        x = np.array(text)
        x_len = len(x)
        if x_len == self.args.batch_size:
            return list(map(id2labels.get, self._predict(sess, x)))
        if x_len < self.args.batch_size:
            x = np.concatenate((x, np.array([x[-1]] * (self.args.batch_size - x_len))))
            return list(map(id2labels.get, self._predict(sess, x)[:len(text)]))
        else:
            n_chunks = x_len / self.args.batch_size
            if x_len % self.args.batch_size:
                n_chunks += 1
            data_list = np.array_split(x[:self.args.batch_size*n_chunks], n_chunks, axis=0)
            results = []
            for m in range(n_chunks):
                results.extend(self.predict_label(sess, id2labels, data_list[m]))
            return results

    def predict_class(self, sess, text):
        x = np.array(text)
        if len(x) != self.args.batch_size:
            x = np.concatenate((x, np.array([x[-1]] * (self.args.batch_size - len(x)))))
            return self._predict(sess, x)[:len(text)]
        else:
            return self._predict(sess, x)
