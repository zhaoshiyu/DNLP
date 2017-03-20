#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import argparse
import pickle

import numpy as np
import tensorflow as tf

from data_loader import TextLoader
from rnn_model import RNNModel
from rnn_model import BIDIRNNModel

class Classifier(object):
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
        self.sess = tf.Session()
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

    def train(self, data_file=None, data=None, dev_data_file=None, vocab_corpus_file=None, args=None, continued=True):
        if self.vocab and self.labels:
            train_data_loader = TextLoader(model_dir=self.args.model_path, 
                                           data_file=data_file, 
                                           vocab_corpus_file=vocab_corpus_file, 
                                           batch_size=self.args.batch_size, 
                                           seq_length=self.args.seq_length,
                                           vocab=self.vocab,
                                           labels=self.labels)
        else:
            train_data_loader = TextLoader(model_dir=self.args.model_path, 
                                           data_file=data_file, 
                                           vocab_corpus_file=vocab_corpus_file, 
                                           batch_size=self.args.batch_size, 
                                           seq_length=self.args.seq_length)
        
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

            # self._load_config()
            # need_be_same = ['model', 'rnn_size', 'num_layers', 'seq_length']
            # for checkme in need_be_same:
            #     assert vars(self.args)[checkme] == vars(self.args)[checkme], 'command line argument and saved model disagree on %s' % checkme
            # import pdb; pdb.set_trace()
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
            # Summaries for loss and accuracy
            loss_summary = tf.summary.scalar("loss", self.model.loss)
            acc_summary = tf.summary.scalar("accuracy", self.model.accuracy)

            # Train Summaries
            train_summary_op = tf.summary.merge([loss_summary, acc_summary])
            train_summary_dir = os.path.join(self.model_path, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, self.sess.graph)

            if dev_data_loader:
                # Dev summaries
                dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
                dev_summary_dir = os.path.join(self.model_path, "summaries", "dev")
                dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, self.sess.graph)

            for epoch in range(self.args.num_epochs):
                self.sess.run(tf.assign(self.model.lr, self.args.learning_rate * (self.args.decay_rate ** epoch)))
                train_data_loader.reset_batch_pointer()
                for batch in range(train_data_loader.num_batches):
                    start = time.time()
                    x, y = train_data_loader.next_batch()
                    feed = {self.model.input_data: x, self.model.targets: y}
                    train_loss, _, accuracy, summaries = self.sess.run([self.model.loss, self.model.optimizer, self.model.accuracy, train_summary_op], feed_dict=feed)
                    end = time.time()
                    print('{}/{} (epoch {}/{}), loss = {:.5f}, accuracy = {:.3f}, time/batch = {:.3f}'\
                        .format(epoch * train_data_loader.num_batches + batch + 1,
                                self.args.num_epochs * train_data_loader.num_batches,
                                epoch + 1,
                                self.args.num_epochs,
                                train_loss,
                                accuracy,
                                end - start))
                    train_summary_writer.add_summary(summaries, epoch * train_data_loader.num_batches + batch + 1)
                    if (epoch * train_data_loader.num_batches + batch + 1) % args.save_every == 0 \
                        or (epoch == args.num_epochs-1 and batch == train_data_loader.num_batches-1):
                        checkpoint_path = os.path.join(self.args.model_path, 'model.ckpt')
                        saver.save(self.sess, checkpoint_path, global_step=epoch * train_data_loader.num_batches + batch + 1)
                        print('model saved to {}'.format(checkpoint_path))

                        if dev_data_loader:
                            x, y = dev_data_loader.next_batch()
                            feed = {self.model.input_data: x, self.model.targets: y}
                            dev_loss, _, dev_accuracy, dev_summaries = self.sess.run([self.model.loss, self.model.optimizer, self.model.accuracy, train_summary_op], feed_dict=feed)
                            print('dev_loss = {:.5f}, dev_accuracy = {:.3f}'.format(dev_loss, dev_accuracy))
                            if dev_summary_writer:
                                dev_summary_writer.add_summary(dev_summaries, epoch * train_data_loader.num_batches + batch + 1) 


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
        data_loader = TextLoader(model_dir=self.args.model_path, 
                                 data_file=test_file, 
                                 batch_size=self.args.batch_size, 
                                 seq_length=self.args.seq_length,
                                 vocab=self.vocab,
                                 labels=self.labels)
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
    parser.add_argument('--rnn_size', type=int, default=128,
                        help='size of RNN hidden state')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='number of layers in RNN')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='minibatch size')
    parser.add_argument('--seq_length', type=int, default=25,
                        help='RNN sequence length')
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='number of epochs')
    parser.add_argument('--save_every', type=int, default=1000,
                        help='save frequency')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--decay_rate', type=float, default=0.9,
                        help='decay rate for rmsprop')
    parser.add_argument('--keep_prob', type=float, default=0.8,
                        help='dropout keep probability')
    parser.add_argument('--state_is_tuple', type=bool, default=True,
                        help='state_is_tuple')
    args = parser.parse_args()
    
    # data = pd.read_csv('../../data/train.csv', encoding='utf-8')
    model_path = '../../data/test-model'
    rnn = Classifier(model_path, args)
    # rnn.train(data_file='../../data/train.csv', dev_data_file='../../data/test.csv', vocab_corpus_file='../../data/corpus.csv', args=args)
    print(rnn.predict(['英超-曼联3-1米堡升至第5 红魔迎来英超600胜']))
    print((rnn.test(test_file='../../data/test.csv', batch_size=32)))
