#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import argparse
import pickle

import numpy as np
import tensorflow as tf

from data_loader import TextLoader
from cnn_model import CNNModel

class CNNClassifier(object):
    def __init__(self, model_path, args):
        assert os.path.isdir(model_path), '%s must be a path' % model_path
        self.model_path = model_path
        self.config_vocab_labels_file = os.path.join(self.model_path, 'config_vocab_labels.pkl')
        self.args = args
        self.args.label_size = None
        self.args.vocab_size = None
        self.vocab = None
        self.labels = None
        self.model = None
        session_conf = tf.ConfigProto(
                                    allow_soft_placement=self.args.allow_soft_placement,
                                    log_device_placement=self.args.log_device_placement)
        self.sess = tf.Session(config=session_conf)
        if os.path.exists(self.config_vocab_labels_file):
            self._load_config()

    def _load_config(self):
        with open(self.config_vocab_labels_file, 'rb') as f:
            saved_args, vocab, labels = pickle.load(f)
        assert saved_args, 'load config error'
        assert vocab, 'load vocab error'
        assert labels, 'load labels error'
        self.args = saved_args
        self.vocab = vocab
        self.labels = labels
        self.id2labels = dict(list(zip(list(labels.values()), list(labels.keys()))))

    def _load_model(self, batch_size=None):
        print('loading model ... ')
        # self.__load_config()
        if batch_size is not None:
            self.args.batch_size = batch_size
        self._init_model()
        saver =tf.train.Saver(tf.global_variables())
        ckpt = tf.train.get_checkpoint_state(self.args.model_path)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(self.sess, ckpt.model_checkpoint_path)

    def _init_model(self):
        # if self.model is None:
        import pdb; pdb.set_trace()
        try:
            with tf.variable_scope("classifier"):
                self.model = CNNModel(
                                seq_length=self.args.seq_length,
                                label_size=self.args.label_size,
                                vocab_size=self.args.vocab_size,
                                embedding_size=self.args.embedding_dim,
                                filter_sizes=list(map(int, self.args.filter_sizes.split(","))),
                                num_filters=self.args.num_filters,
                                l2_reg_lambda=self.args.l2_reg_lambda)
                # self.model = BIDIRNNModel(self.args)
        except ValueError as ve:
            with tf.variable_scope("classifier", reuse=True):
                self.model = CNNModel(
                                seq_length=seq_length,
                                label_size=self.args.label_size,
                                vocab_size=self.args.vocab_size,
                                embedding_size=self.args.embedding_dim,
                                filter_sizes=list(map(int, self.args.filter_sizes.split(","))),
                                num_filters=self.args.num_filters,
                                l2_reg_lambda=self.args.l2_reg_lambda)
                # self.model = BIDIRNNModel(self.args)

    def _transform(self, text):
        text = text if type('') == type(text) else text.decode('utf-8')
        x = list(map(self.vocab.get, text))
        x = [i if i else 0 for i in x]
        if len(x) >= self.args.seq_length:
            x = x[:self.args.seq_length]
        else:
            x = x + [0] * (self.args.seq_length - len(x))
        return x

    def train(self, data_file=None, data=None, dev_data_file=None, vocab_corpus_file=None, args=None, continued=True):
        train_data_loader = TextLoader(model_dir=self.args.model_path, 
                                       data_file=data_file, 
                                       vocab_corpus_file=vocab_corpus_file, 
                                       batch_size=self.args.batch_size, 
                                       seq_length=self.args.seq_length,
                                       vocab=self.vocab,
                                       labels=self.labels)
        
        if dev_data_file:
            if self.vocab and self.labels:
                vocab = self.vocab
                labels = self.labels
            else:
                vocab = train_data_loader.vocab
                labels = train_data_loader.labels
            dev_data_loader = TextLoader(model_dir=self.args.model_path, 
                                        data_file=data_file, 
                                        batch_size=self.args.batch_size, 
                                        seq_length=self.args.seq_length,
                                        vocab=vocab,
                                        labels=labels)

        if not self.args.vocab_size and not self.args.label_size:
            self.args.vocab_size = train_data_loader.vocab_size
            self.args.label_size = train_data_loader.label_size
        
        self._init_model()
        init = tf.global_variables_initializer()
        self.sess.run(init)
        saver = tf.train.Saver(tf.global_variables())

        if os.path.isfile(self.config_vocab_labels_file) and continued:
            ckpt = tf.train.get_checkpoint_state(self.args.model_path)
            assert ckpt, 'No checkpoint found'
            assert ckpt.model_checkpoint_path, 'No model path found in checkpoint'
            with open(self.config_vocab_labels_file, 'rb') as f:
                saved_args, vocab, labels = pickle.load(f)
            need_be_same = ['model', 'rnn_size', 'num_layers', 'seq_length']
            for checkme in need_be_same:
                assert vars(saved_args)[checkme] == vars(self.args)[checkme], 'command line argument and saved model disagree on %s' % checkme
            assert len(self.vocab) == len(train_data_loader.vocab), 'data and loaded model disagree on dictionary mappings'
            assert len(self.labels) == len(train_data_loader.labels), 'data and loaded model disagree on label dictionary mappings'
            print('loading last training model and continue')
            saver.restore(self.sess, ckpt.model_checkpoint_path)
        else:
            self.vocab = train_data_loader.vocab
            self.labels = train_data_loader.labels
            self.args.vocab_size = train_data_loader.vocab_size
            self.args.label_size = train_data_loader.label_size
            with open(self.config_vocab_labels_file, 'wb') as f:
                pickle.dump([self.args, self.vocab, self.labels], f)

        with tf.Graph().as_default():
            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(1e-3)
            grads_and_vars = optimizer.compute_gradients(cnn.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            # Keep track of gradient values and sparsity (optional)
            grad_summaries = []
            for g, v in grads_and_vars:
                if g is not None:
                    grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                    sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                    grad_summaries.append(grad_hist_summary)
                    grad_summaries.append(sparsity_summary)
            grad_summaries_merged = tf.summary.merge(grad_summaries)

            # Output directory for models and summaries
            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
            print("Writing to {}\n".format(out_dir))

            # Summaries for loss and accuracy
            loss_summary = tf.summary.scalar("loss", cnn.loss)
            acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

            # Train Summaries
            train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            # Dev summaries
            dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
            dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
            dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

            # Write vocabulary
            # vocab_processor.save(os.path.join(out_dir, "vocab"))

            # Initialize all variables
            sess.run(tf.global_variables_initializer())

            def make_y_batch(y_batch):
                y_batch = np.asarray([np.argmax(y_i) for y_i in y_batch])

            def train_step(x_batch, y_batch):
                """
                A single training step
                """
                # y_batch = np.asarray([np.argmax(y_i) for y_i in y_batch])
                feed_dict = {
                cnn.input_x: x_batch,
                cnn.input_y: y_batch,
                cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
                }
                
                _, step, summaries, loss, accuracy = sess.run(
                    [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                train_summary_writer.add_summary(summaries, step)

            def dev_step(x_batch, y_batch, writer=None):
                """
                Evaluates model on a dev set
                """
                # y_batch = np.asarray([np.argmax(y_i) for y_i in y_batch])
                feed_dict = {
                cnn.input_x: x_batch,
                cnn.input_y: y_batch,
                cnn.dropout_keep_prob: 1.0
                }
                step, summaries, loss, accuracy = sess.run(
                    [global_step, dev_summary_op, cnn.loss, cnn.accuracy],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                if writer:
                    writer.add_summary(summaries, step)

            # Generate batches
            train_data_loader.reset_batch_pointer()
            # Training loop. For each batch...
            for batch in range(train_data_loader.num_batches):
                x_batch, y_batch = train_data_loader.next_batch()
                train_step(x_batch, y_batch)
                current_step = tf.train.global_step(sess, global_step)
                if current_step % self.args.evaluate_every == 0:
                    print("\nEvaluation:")
                    # dev_step(x_dev, y_dev, writer=dev_summary_writer)
                    dev_step(x_batch, y_batch, writer=dev_summary_writer)
                    print("")
                if current_step % self.args.checkpoint_every == 0:
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print("Saved model checkpoint to {}\n".format(path))



# class Config(object):
#     model_path = '../../data/test-model'
#     train_file = '../../data/input.csv'
#     vocab_corpus_file = '../../data/corpus.txt'
#     init_from = None
#     model = 'lstm'
#     state_is_tuple = True
#     learning_rate = 0.001
#     decay_rate = 0.9
#     keep_prob = 0.8
#     rnn_size = 64
#     num_layers = 2
#     seq_length = 20
#     batch_size = 16
#     num_epochs = 20
#     num_epochs = 50
#     save_every = 100
#     vocab_size = None
#     label_size = None

if __name__ == '__main__':
    # rnn_classifier_train_test()
    # cnn_classifier_train_test()
    model_path = '../../data/test-model-cnn'

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default= model_path,
                        help='directory to store checkpointed models')

    parser.add_argument('--dev_sample_percentage', type=float, default= .1,
                        help='Percentage of the training data to use for validation')
    
    parser.add_argument('--embedding_dim', type=int, default= 128,
                        help='Dimensionality of character embedding (default: 128)')
    
    parser.add_argument('--cnn_size', type=int, default=128,
                        help='size of CNN hidden state')
    
    parser.add_argument('--filter_sizes', type=str, default= '3,4,5',
                        help='Comma-separated filter sizes (default: "3,4,5")')

    parser.add_argument('--num_filters', type=int, default= 128,
                        help='Number of filters per filter size (default: 128)')
    
    parser.add_argument('--dropout_keep_prob', type=float, default= 0.5,
                        help='Dropout keep probability (default: 0.5)')

    parser.add_argument('--l2_reg_lambda', type=float, default= 0.0,
                        help='L2 regularization lambda (default: 0.0)')

    parser.add_argument('--batch_size', type=int, default=64,
                        help='minibatch size (default: 64)')
    parser.add_argument('--seq_length', type=int, default=25,
                        help='sequence length  (default: 25)')
    parser.add_argument('--num_epochs', type=int, default=200,
                        help='Number of training epochs (default: 200)')

    

    parser.add_argument('--evaluate_every', type=int, default=100,
                        help='Evaluate model on dev set after this many steps (default: 100)')
    parser.add_argument('--checkpoint_every', type=int, default=100,
                        help='Save model after this many steps (default: 100)')
    parser.add_argument('--num_checkpoints', type=int, default=5,
                        help='Number of checkpoints to store (default: 5)')


    parser.add_argument('--allow_soft_placement', type=bool, default=True,
                        help='Allow device soft device placement')
    parser.add_argument('--log_device_placement', type=bool, default=False,
                        help='Log placement of ops on devices')
    parser.add_argument('--label_size', type=int, default=4,
                        help='Classes number')
    
    args = parser.parse_args()
    
    # data = pd.read_csv('../../data/train.csv', encoding='utf-8')
    model_path = '../../data/test-model-cnn'
    cnn = CNNClassifier(model_path, args)
    cnn.train(data_file='../../data/train.csv', dev_data_file='../../data/test.csv', vocab_corpus_file='../../data/corpus.csv', args=args)
    print(cnn.predict(['英超-曼联3-1米堡升至第5 红魔迎来英超600胜']))
    print((cnn.test(test_file='../../data/test.csv', batch_size=32)))