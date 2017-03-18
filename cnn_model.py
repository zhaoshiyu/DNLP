# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np

class CNNModel(object):
    """
    CNN for text classification.
    embedding layer, followed by convolutional, max-pooling and softmax layer
    """
    def __init__(
      self, sequence_length, label_size, vocab_size,
      embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0):
        self.seq_length = sequence_length
        self.label_size = label_size
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.filter_sizes = filter_sizes
        self.num_filters = num_filters
        self.l2_reg_lambda = l2_reg_lambda
        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int64, [None, self.seq_length], name="input_x")
        self.input_y = tf.placeholder(tf.int64, [None, ], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            self.W = tf.Variable(
                tf.random_uniform([self.vocab_size, self.embedding_size], -1.0, 1.0),
                name="W")
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)
        
        # with tf.variable_scope('embeddingLayer'):
        #     with tf.device('/cpu:0'):
        #         self.W = tf.get_variable('W', [self.vocab_size, self.embedding_size])
        #         # embedded = tf.nn.embedding_lookup(self.W, self.input_x)
        #         self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
        #         self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)
        #         # inputs = tf.split(embedded, self.seq_length, 1)
        #         # inputs = [tf.squeeze(input_, [1]) for input_ in inputs]

        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(self.filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, self.embedding_size, 1, self.num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[self.num_filters]), name="b")
                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, self.seq_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = self.num_filters * len(self.filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        # with tf.name_scope("output"):
        #     W = tf.get_variable(
        #         "W",
        #         shape=[num_filters_total, self.label_size],
        #         initializer=tf.contrib.layers.xavier_initializer())
        #     b = tf.Variable(tf.constant(0.1, shape=[self.label_size]), name="b")
        #     l2_loss += tf.nn.l2_loss(W)
        #     l2_loss += tf.nn.l2_loss(b)
        #     self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
        #     logits = self.scores
        #     self.predictions = tf.argmax(self.scores, 1, name="predictions")
        
        with tf.variable_scope('softmaxLayer'):
            softmax_w = tf.get_variable('w', [num_filters_total, self.label_size])
            softmax_b = tf.get_variable('b', [self.label_size])
            l2_loss += tf.nn.l2_loss(softmax_w)
            l2_loss += tf.nn.l2_loss(softmax_b)
            logits = tf.matmul(self.h_drop, softmax_w) + softmax_b
            self.predictions = tf.argmax(logits, 1, name="predictions")
            # self.predictions = tf.nn.softmax(logits)

        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
            self.loss = tf.reduce_mean(tf.contrib.nn.deprecated_flipped_sparse_softmax_cross_entropy_with_logits(logits=logits, labels=self.input_y))
            # losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=self.input_y)
            # self.loss = tf.reduce_mean(losses) + self.l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            # correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            # correct_predictions = tf.equal(tf.argmax(self.predictions, 1), self.input_y)
            correct_predictions = tf.equal(self.predictions, self.input_y)
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
