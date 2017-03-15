from SentenceGramMachine.BaseMachine import *
import tensorflow as tf
from tensorflow.models.rnn import rnn_cell
from six.moves import xrange
import numpy as np

class SentenceGramMachineV2(BaseMachine):
    def __init__(self):
        BaseMachine.__init__(self)

    def InitConfig(self, block_steps=5, block_num = 1000):
        self.embedding_size = len(self.reverse_dictionary[1][BaseMachine.VECTOR])
        self.vectors = np.zeros((self.length + 1, self.embedding_size), dtype=np.float32)
        for i in xrange(self.length):
            self.vectors[i + 1, :] = self.reverse_dictionary[i + 1][BaseMachine.VECTOR]
        # self.session = tf.Session()
        # self.session.run(tf.initialize_all_variables())
        self.block_steps = block_steps
        self.block_num = block_num

    def DrawGraph(self):
        graph = tf.Graph()
        with graph.as_default():
            train_input = tf.placeholder(tf.int32, shape=[1])
            train_label = tf.placeholder(tf.float32, shape=[1, 1])
            embeddings = tf.constant(self.vectors)
            word_embed = tf.nn.embedding_lookup(embeddings, train_input)
            lstm = rnn_cell.BasicLSTMCell(self.embedding_size)
            state = tf.zeros([1, lstm.state_size], tf.float32)
            # self.state = (1.0 - self.train_label) * self.state
            # tf.get_variable_scope().reuse_variables()
            output, state = lstm(word_embed, state)
            weight = tf.Variable(tf.random_uniform([self.embedding_size, 1], -1.0, 1.0))
            bias = tf.Variable(tf.zeros([1]))
            y = tf.nn.sigmoid(tf.matmul(output, weight) + bias)
            loss = tf.abs(train_label - y[0, 0])
            optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

    def Training(self):
        average_loss = 0
        sentenceList = self.nextTrainSentences()

        cnt = 0
        for sentenceItem in sentenceList:
            for i in range(len(sentenceItem)):
                if i < len(sentenceItem) - 1:
                    label = 0
                else:
                    label = 1
                _, loss = self.session.run([self.optimizer, self.loss], feed_dict={
                    self.train_input: sentenceItem[i],
                    self.train_label: np.array(label, ndmin=2),
                })
            average_loss += loss
            cnt += 1
        average_loss /= cnt
        print("Average loss at block:", average_loss)
        return average_loss

    def TrainModal(self):
        for step in range(self.block_steps):
            average_loss = 0
            for _ in range(self.block_num):
                average_loss += self.Training()
            average_loss /= self.block_num
            print("Average loss at step ", step, ": ", average_loss)
