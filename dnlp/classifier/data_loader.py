# -*- coding: utf-8 -*-

import os
import collections

import pickle
import numpy as np
import pandas as pd

from trans import Transfer


class TextLoader(object):
    def __init__(self, 
        model_dir=None, 
        data_file=None, 
        data = None,
        vocab_corpus_file=None, 
        batch_size=None, 
        seq_length=None, 
        vocab=None,
        labels=None,
        encoding='utf8'):
        self.data_file = data_file
        self.vocab_corpus_file = vocab_corpus_file
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.encoding = encoding
        
        self.data = pd.read_csv(self.data_file, encoding=encoding) if self.data_file and os.path.exists(self.data_file) else data
        # if self.data_file and os.path.exists(self.data_file):
        #     self.data = pd.read_csv(self.data_file, encoding=encoding)
        # else:
        #     print('data file not exists, please specify training file ')
        #     exit(1)
        if self.data is None:
            print('data is null, please specify training file ')
            exit(1)
        if model_dir:
            self.model_dir = model_dir
            self.labels = labels if labels is not None else self.preprocess_labels(self.data['label'])
            if vocab is not None:
                self.vocab = vocab
            elif os.path.exists(self.vocab_corpus_file):
                print('reading corpus and processing vocab')
                self.vocab = self.preprocess_vocab_file(self.vocab_corpus_file)
            else:
                print('processing vocab by data')
                corpus = self.data['text'].values.tolist()
                corpus.extend(self.data['label'].values.tolist())
                corpus = ' '.join(corpus)
                self.vocab = self.preprocess_vocab(corpus)
            
            assert self.vocab, 'vocab is null'
            assert self.labels, 'labels is null'
            self.label_size = len(self.labels)
            self.vocab_size = len(self.vocab) + 1

            # labels = set(self.data['label'])
            # if os.path.exists(self.vocab_corpus_file):
            #     print('reading corpus and processing vocab')
            #     self.transfer = Transfer(self.model_dir, self.seq_length, 
            #                             label_data=labels, vocab_corpus_file=self.vocab_corpus_file)
            # else:
            #     print('processing vocab by training file ')
            #     self.transfer = Transfer(self.model_dir, self.seq_length, labels, self.data_file)

            '''
            with open(self.vocab_file, 'wb') as f:
                pickle.dump(self.vocab, f)

            self.label_file = os.path.join(model_dir, 'labels.pkl')
            self.vocab_file = os.path.join(model_dir, 'vocab.pkl')

            if os.path.exists(self.label_file):
                with open(self.label_file, 'rb') as f:
                    self.labels = pickle.load(f)
            else:
                self.labels = self.preprocess_labels(self.data)
                with open(self.label_file, 'wb') as f:
                    pickle.dump(self.labels, f)
            self.label_size = len(self.labels)
            
            if os.path.exists(self.vocab_file):
                with open(self.vocab_file, 'rb') as f:
                    self.vocab = pickle.load(f)
            else:
                if os.path.exists(self.vocab_corpus_file):
                    print('reading corpus and processing vocab')
                    self.vocab = self.preprocess_vocab_file(self.vocab_corpus_file)
                else:
                    print('processing vocab by data')
                    self.vocab = self.preprocess_vocab(self.data)
                with open(self.vocab_file, 'wb') as f:
                    pickle.dump(self.vocab, f)
            self.vocab_size = len(self.vocab) + 1
            '''

            self.tensor = self.preprocess_tensor(self.data)

        self.reset_batch_pointer()

    def preprocess_labels(self, labels_data):
        count = 0
        labels = {}
        for label in set(labels_data):
            labels[label] = count
            count += 1
        return labels

    def preprocess_vocab(self, corpus):
        counter = collections.Counter(corpus)
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
    
    def preprocess_tensor(self, data):
        tensor_x = np.array(list(map(self.transform, data['text'])))
        tensor_y = np.array(list(map(self.labels.get, data['label'])))
        return np.c_[tensor_x, tensor_y].astype(int)

    def transform(self, text):
        vector_ids = list(map(self.vocab.get, text))
        vector_ids = [i if i else 0 for i in vector_ids]
        if len(vector_ids) >= self.seq_length:
            vector_ids = vector_ids[:self.seq_length]
        else:
            vector_ids = vector_ids + [0] * (self.seq_length - len(vector_ids))
        return vector_ids

    def create_batches(self):
        self.num_batches = int(self.tensor.shape[0] / self.batch_size)
        if self.num_batches == 0:
            assert False, 'Not enough data, make batch_size small.'

        np.random.shuffle(self.tensor)
        tensor = self.tensor[:self.num_batches * self.batch_size]
        self.x_batches = np.split(tensor[:, :-1], self.num_batches, 0)
        self.y_batches = np.split(tensor[:, -1], self.num_batches, 0)


    def next_batch(self):
        x = self.x_batches[self.pointer]
        y = self.y_batches[self.pointer]
        self.pointer += 1
        return x, y


    def reset_batch_pointer(self):
        self.create_batches()
        self.pointer = 0

if __name__ == '__main__':
    print('111')
    data_loader = TextLoader(model_dir='../../data/test-model', data_file='../../data/input.csv', vocab_corpus_file='../../data/corpus.txt', data=None, batch_size=32, seq_length=30)
    print(data_loader.vocab)
    import pdb; pdb.set_trace()
    data_loader.reset_batch_pointer()
    for batch in range(data_loader.num_batches):
        x, y = data_loader.next_batch()
        print(x)
        print(y)
