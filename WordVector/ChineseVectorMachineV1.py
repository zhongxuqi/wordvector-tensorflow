from WordVector.BaseVectorTrainer import BaseVectorMachine
from langPreprocesser.ChinesePreprocesser import Article, Sentence
import tensorflow as tf
from tensorflow.models.rnn import rnn_cell
from six.moves import xrange
import numpy as np
import Util

class ChineseVectorMachine(BaseVectorMachine):
    def GenerateBatch(self, batch_size):
        batch = np.ndarray(shape=(batch_size), dtype=np.int32)
        labels = np.ndarray(shape=(batch_size), dtype=np.int32)
        for i in range(batch_size):
            prevWord = self._nextWord()
            while (prevWord != None) and (not Util.HasChineseWord(prevWord)):
                prevWord = self._nextWord()
            nextWord = self._nextWord()
            while (nextWord != None) and (not Util.HasChineseWord(nextWord)):
                nextWord = self._nextWord()
            if (prevWord != None) and (nextWord != None):
                if prevWord in self.dictionary:
                    batch[i] = self.dictionary[prevWord]["index"]
                else:
                    batch[i] = 0
                if nextWord in self.dictionary:
                    labels[i] = self.dictionary[nextWord]["index"]
                else:
                    labels[i] = 0
            else:
                break
        return batch, labels

    def InitConfig(self, batch_size=1024, embedding_size=256, train_steps=128, \
    iscontinue=False):
        self.batch_size = batch_size
        self.embedding_size = embedding_size
        self.train_steps = train_steps
        self.vectors = None
        if iscontinue and ("vector" in self.reverse_dictionary[1]):
            self.embedding_size = len(self.reverse_dictionary[1]["vector"])
            self.vectors = np.zeros([self.length + 1, \
            self.embedding_size], np.float32)
            self.vectors[0, :] = np.random.random(self.embedding_size)
            for i in xrange(self.length):
                if "vector" in self.reverse_dictionary[i + 1]:
                    self.vectors[i + 1, :] = np.array(self.reverse_dictionary[i + 1]["vector"])
                else:
                    self.vectors[i + 1, :] = np.random.random(self.embedding_size)

    def DrawGraph(self):
        graph = tf.Graph()
        with graph.as_default():
            data_input = tf.placeholder(tf.int32, shape=[1])
            labels_input = tf.placeholder(tf.int32, shape=[1])

            if self.vectors is None:
                embeddings = tf.Variable(tf.random_uniform([self.length + 1, \
                self.embedding_size], -1.0, 1.0))
            else:
                embeddings = tf.Variable(self.vectors)
            data_embed = tf.nn.embedding_lookup(embeddings, data_input)
            labels_embed = tf.nn.embedding_lookup(embeddings, labels_input)

            # 1th layer
            LSTM_1 = rnn_cell.BasicLSTMCell(256)
            # Initial state of the LSTM memory.
            state1 = tf.zeros([1, LSTM_1.state_size], tf.float32)
            output_1, state1 = LSTM_1(data_embed, state1, scope="rnn_layer1")

            # 2th layer
            # LSTM_2 = rnn_cell.BasicLSTMCell(256)
            # # Initial state of the LSTM memory.
            # state2 = tf.zeros([1, LSTM_2.state_size], tf.float32)
            # output_2, state2 = LSTM_2(output_1, state2, scope="rnn_layer2")
            #
            # # 3th layer
            # LSTM_3 = rnn_cell.BasicLSTMCell(256)
            # # Initial state of the LSTM memory.
            # state3 = tf.zeros([1, LSTM_3.state_size], tf.float32)
            # output_3, state3 = LSTM_2(output_2, state3, scope="rnn_layer3")

            loss = 1 - tf.reduce_sum(tf.mul(output_1, labels_embed)) / \
            tf.sqrt(tf.reduce_sum(tf.square(output_1))) / \
            tf.sqrt(tf.reduce_sum(tf.square(labels_embed)))
            optimizer = tf.train.GradientDescentOptimizer(2.0).minimize(loss)
        return graph, data_input, labels_input, loss, optimizer, embeddings

    def Training(self, graph, data_input, labels_input, loss, optimizer, embeddings):
        with tf.Session(graph=graph) as session:
            # We must initialize all variables before we use them.
            tf.initialize_all_variables().run()
            step = 0
            while (self.train_steps < 0) or (step < self.train_steps):
                batch_inputs, batch_labels = self.GenerateBatch(self.batch_size)
                average_loss = 0
                for i in xrange(len(batch_inputs)):
                    feed_dict = {data_input : batch_inputs[i:(i+1)], \
                    labels_input : batch_labels[i:(i+1)]}
                    _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
                    average_loss += loss_val
                if len(batch_inputs) > 0:
                    average_loss /= self.batch_size
                    # The average loss is an estimate of the loss over the last 2000 batches.
                    print("Average loss at step ", step, ": ", average_loss)
                else:
                    break
                step += 1
            self.SaveModal(embeddings)

    def SaveModal(self, embeddings):
        self.vectors = embeddings.eval()

    def StartTrain(self):
        graph, data_input, labels_input, loss, optimizer, embeddings = self.DrawGraph()
        self.Training(graph, data_input, labels_input, loss, optimizer, embeddings)
