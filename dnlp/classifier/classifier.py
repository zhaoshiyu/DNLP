#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import argparse
import pickle

import pandas as pd

import numpy as np
import tensorflow as tf

from data_loader import TextLoader
from rnn_model import RNNModel
from rnn_model import BIDIRNNModel

class Classifier(object):
    def __init__(self, model_path, args):
        assert os.path.isdir(model_path), '%s must be a path' % model_path
        self.model_path = model_path
        self.config_file = os.path.join(self.model_path, 'config.pkl')
        self.args = args
        self.transfer = None
        self.model = None
        self.sess = tf.Session()
        if os.path.exists(self.config_file):
            self._load_config()

    def _load_config(self):
        with open(self.config_file, 'rb') as f:
            saved_args = pickle.load(f)
        assert saved_args, 'load config error'
        self.args = saved_args
        with open(os.path.join(self.model_path, 'test.pkl'), 'rb') as f:
            saved_args, vocab, labels = pickle.load(f)
        import pdb; pdb.set_trace()
        print(labels)

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
        try:
            with tf.variable_scope("classifier"):
                self.model = RNNModel(self.args)
                # self.model = BIDIRNNModel(self.args, deterministic=deterministic)
        except ValueError as ve:
            with tf.variable_scope("classifier", reuse=True):
                self.model = RNNModel(self.args)
                # self.model = BIDIRNNModel(self.args, deterministic=deterministic)

    def _transform(self, text):
        text = text if type('') == type(text) else text.decode('utf-8')
        x = list(map(self.vocab.get, text))
        x = [i if i else 0 for i in x]
        if len(x) >= self.args.seq_length:
            x = x[:self.args.seq_length]
        else:
            x = x + [0] * (self.args.seq_length - len(x))
        return x

    def load(self):
        self.close()
        self._load_model()

    def close(self):
        self.args = None
        self.vocab = None
        self.labels = None
        self.id2labels = None
        self.model = None
        if self.sess:
            self.sess.close()
        self.sess = None

    def train(self, data_file=None, data=None, vocab_corpus_file=None, args=None, continued=False):
        data_loader = TextLoader(model_dir=self.args.model_path, data_file=data_file, vocab_corpus_file=vocab_corpus_file, batch_size=self.args.batch_size, seq_length=self.args.seq_length)
        self.transfer = data_loader.transfer
        self.args.vocab_size = data_loader.transfer.vocab_size
        self.args.label_size = data_loader.transfer.label_size

        with open(self.config_file, 'wb') as f:
            pickle.dump(self.args, f)
        with open(os.path.join(self.model_path, 'test.pkl'), 'wb') as f:
            pickle.dump([self.args, self.transfer.vocab, self.transfer.labels], f)

        self._init_model()
        init = tf.global_variables_initializer()
        self.sess.run(init)
        saver = tf.train.Saver(tf.global_variables())

        # import pdb; pdb.set_trace()
        if os.path.isfile(self.config_file) and continued:
            ckpt = tf.train.get_checkpoint_state(self.args.model_path)
            assert ckpt, 'No checkpoint found'
            assert ckpt.model_checkpoint_path, 'No model path found in checkpoint'

            self._load_config()
            need_be_same = ['model', 'rnn_size', 'num_layers', 'seq_length']
            for checkme in need_be_same:
                assert vars(self.args)[checkme] == vars(self.args)[checkme], 'command line argument and saved model disagree on %s' % checkme

            assert self.vocab == data_loader.vocab, 'data and loaded model disagree on dictionary mappings'
            assert self.labels == data_loader.labels, 'data and loaded model disagree on label dictionary mappings'

            print('loading last training model and continue')
            saver.restore(self.sess, ckpt.model_checkpoint_path)
        
        with tf.Graph().as_default():
            # Summaries for loss and accuracy
            loss_summary = tf.summary.scalar("loss", self.model.loss)
            acc_summary = tf.summary.scalar("accuracy", self.model.accuracy)

            # Train Summaries
            train_summary_op = tf.summary.merge([loss_summary, acc_summary])
            train_summary_dir = os.path.join(self.model_path, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, self.sess.graph)

            for epoch in range(self.args.num_epochs):
                self.sess.run(tf.assign(self.model.lr, self.args.learning_rate * (self.args.decay_rate ** epoch)))
                data_loader.reset_batch_pointer()
                for batch in range(data_loader.num_batches):
                    start = time.time()
                    x, y = data_loader.next_batch()
                    feed = {self.model.input_data: x, self.model.targets: y}
                    train_loss, _, accuracy, summaries = self.sess.run([self.model.loss, self.model.optimizer, self.model.accuracy, train_summary_op], feed_dict=feed)
                    end = time.time()
                    print('{}/{} (epoch {}/{}), loss = {:.5f}, accuracy = {:.3f}, time/batch = {:.3f}'\
                        .format(epoch * data_loader.num_batches + batch + 1,
                                self.args.num_epochs * data_loader.num_batches,
                                epoch + 1,
                                self.args.num_epochs,
                                train_loss,
                                accuracy,
                                end - start))
                    train_summary_writer.add_summary(summaries, epoch * data_loader.num_batches + batch + 1)
                    if (epoch * data_loader.num_batches + batch + 1) % args.save_every == 0 \
                        or (epoch == args.num_epochs-1 and batch == data_loader.num_batches-1):
                        checkpoint_path = os.path.join(self.args.model_path, 'model.ckpt')
                        saver.save(self.sess, checkpoint_path, global_step=epoch * data_loader.num_batches + batch + 1)
                        print('model saved to {}'.format(checkpoint_path))


    def predict(self, contents, batch_size=32):
        if self.model is None or self.args is None or self.args.batch_size != batch_size or self.vocab is None or self.sess is None or self.id2labels is None:
            self._load_model(batch_size=batch_size)
        x = [self._transform(i.strip()) for i in contents]
        x_len = len(x)
        n_chunks = x_len // self.args.batch_size
        if x_len % self.args.batch_size:
            n_chunks += 1
        x = np.array_split(x[:self.args.batch_size*n_chunks], n_chunks, axis=0)
        results = []
        for m in range(n_chunks):
            results.extend(self.model.predict_label(self.sess, self.id2labels, x[m]))
        return results

    def test(self, test_file=None, data=None, batch_size=32):
        if self.model is None or self.args is None or self.args.batch_size != batch_size or self.vocab is None or self.sess is None or self.id2labels is None or self.labels is None:
            self._load_model(batch_size=batch_size)
        data_loader = TextLoader(model_dir=self.args.model_path, data_file=test_file, batch_size=self.args.batch_size, seq_length=self.args.seq_length)
        data = data_loader.tensor.copy()
        n_chunks = len(data) // self.args.batch_size
        if len(data) % self.args.batch_size:
            n_chunks += 1
        data_list = np.array_split(data[:self.args.batch_size*n_chunks], n_chunks, axis=0)
        correct_total = 0.0
        num_total = 0.0
        for m in range(n_chunks):
            start = time.time()
            x = data_list[m][:, :-1]
            y = data_list[m][:, -1]
            results = self.model.predict_class(self.sess, x)
            correct_num = np.sum(results==y)
            end = time.time()
            print(('batch {}/{} time = {:.3f}, sub_accuracy = {:.6f}'.format(m+1, n_chunks, end-start, correct_num*1.0/len(x))))

            correct_total += correct_num
            num_total += len(x)

        accuracy_total = correct_total / num_total
        print(('total_num = {}, total_accuracy = {:.6f}'.format(int(num_total), accuracy_total)))
        return accuracy_total

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
    
    model_path = '../../data/test-model'

    parser = argparse.ArgumentParser()

    parser.add_argument('--model_path', type=str, default= model_path,
                        help='directory to store checkpointed models')

    parser.add_argument('--model', type=str, default='lstm',
                        help='rnn, gru or lstm, default lstm')

    parser.add_argument('--rnn_size', type=int, default=64,
                        help='size of RNN hidden state')

    parser.add_argument('--num_layers', type=int, default=2,
                        help='number of layers in RNN')

    parser.add_argument('--batch_size', type=int, default=16,
                        help='minibatch size')

    parser.add_argument('--seq_length', type=int, default=20,
                        help='RNN sequence length')

    parser.add_argument('--num_epochs', type=int, default=50,
                        help='number of epochs')

    parser.add_argument('--save_every', type=int, default=1000,
                        help='save frequency')

    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='learning rate')

    parser.add_argument('--decay_rate', type=float, default=0.9,
                        help='decay rate for rmsprop')

    parser.add_argument('--keep_prob', type=float, default=0.8,
                        help='dropout keep probability')

    #parser.add_argument('--stop_loss', type=float, default=0.00001,
    #                    help='training stop loss value')

    parser.add_argument('--state_is_tuple', type=bool, default=True,
                        help='state_is_tuple')
    args = parser.parse_args()
    # config = parser.parse_args()
    # import pdb; pdb.set_trace()
    model_path = '../../data/test-model'
    data = pd.read_csv('../../data/input.csv', encoding='utf-8')
    rnn = Classifier(model_path, args)
    # rnn.train(data_file='../../data/input.csv', vocab_corpus_file='../../data/corpus.txt', args=args)
    print((rnn.test(test_file='../../data/test.csv', batch_size=5000)))
