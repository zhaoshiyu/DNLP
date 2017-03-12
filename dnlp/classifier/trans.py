# -*- coding: utf-8 -*-

import os

import collections
import numpy as np

import pickle

def text_to_tensor(self, vocab_dict, text, seq_length):
    vector_ids = list(map(vocab_dict.get, text[:seq_length]))
    vector_ids = [i if i else 0 for i in vector_ids]
    if len(vector_ids) >= seq_length:
        vector_ids = vector_ids[:seq_length]
    else:
        vector_ids = vector_ids + [0] * (seq_length - len(vector_ids))
    return vector_ids

def label_to_tensor(self, labels_dict, text_labels):
    return np.array(list(map(labels_dict.get, text_labels)))

def tensor_to_label(self, id2labels, label_ids):
    return list(map(id2labels.get, label_ids))



class Transfer(object):
    def __init__(self, data_dir, seq_length, label_data=None, vocab_corpus_file=None):
        self.seq_length = seq_length
        self.vocab_labels_file = os.path.join(data_dir, 'vocab_labels.pkl')
        if os.path.exists(self.vocab_labels_file):
            with open(self.vocab_labels_file, 'rb') as f:
                self.vocab, self.labels = pickle.load(f)
        elif label_data and vocab_corpus_file:
            self.labels = self.preprocess_labels(label_data)
            assert os.path.isfile(vocab_corpus_file), '%s file does not exist' % vocab_corpus_file
            self.vocab = self.preprocess_vocab_file(vocab_corpus_file)
            with open(self.vocab_labels_file, 'wb') as f:
                pickle.dump([self.vocab, self.labels], f)
        else:
            print('label data is null or not vecab corpus file, please check and try again')
            exit(1)
        assert self.vocab, 'vocab is null'
        assert self.labels, 'labels is null'
        self.label_size = len(self.labels)
        self.vocab_size = len(self.vocab) + 1
        self.id2labels = dict(list(zip(list(self.labels.values()), list(self.labels.keys()))))

    def preprocess_labels(self, labels_data):
        count = 0
        labels = {}
        for label in set(labels_data):
            labels[label] = count
            count += 1
        return labels

    def preprocess_vocab(self, data):
        counter = collections.Counter(data)
        count_pairs = sorted(list(counter.items()), key=lambda i: -i[1])
        chars, _ = list(zip(*count_pairs))
        return dict(list(zip(chars, list(range(1, len(chars)+1)))))
        
    def preprocess_vocab_file(self, vocab_corpus_file):
        if not os.path.exists(vocab_corpus_file):
            print('not vocab corpus file')
            exit(1)
        with open(vocab_corpus_file, 'r') as f:
            corpus = f.readlines()
            corpus = ' '.join([i.strip() for i in corpus])
            # corpus = corpus.decode('utf8')
        return self.preprocess_vocab(corpus)

    def transform(self, text):
        vector_ids = list(map(self.vocab.get, text))
        vector_ids = [i if i else 0 for i in vector_ids]
        if len(vector_ids) >= self.seq_length:
            vector_ids = vector_ids[:self.seq_length]
        else:
            vector_ids = vector_ids + [0] * (self.seq_length - len(vector_ids))
        return vector_ids

    def text_to_tensor(self, text):
        vector_ids = list(map(self.vocab.get, text[:self.seq_length]))
        vector_ids = [i if i else 0 for i in vector_ids]
        if len(vector_ids) >= self.seq_length:
            vector_ids = vector_ids[:self.seq_length]
        else:
            vector_ids = vector_ids + [0] * (self.seq_length - len(vector_ids))
        return vector_ids

    def label_to_tensor(self, text_labels):
        return np.array(list(map(self.labels.get, text_labels)))

    def tensor_to_label(self, label_tensor):
        return list(map(self.id2labels.get, label_tensor))
             

