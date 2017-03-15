from SentenceGramMachine.BaseMachine import *
import tensorflow as tf
from six.moves import xrange
import numpy as np
import math

class SentenceGramMachineV1(BaseMachine):
    MODEL_FILENAME = "SentenceGramMachineV1.model"

    def __init__(self):
        BaseMachine.__init__(self)

    def InitConfig(self, block_steps=-1, block_num = 1000):
        self.embedding_size = len(self.reverse_dictionary[1][SentenceGramMachineV1.VECTOR])
        self.vectors = np.zeros((self.length + 1, self.embedding_size), dtype=np.float32)
        for i in xrange(self.length):
            self.vectors[i + 1, :] = self.reverse_dictionary[i + 1][BaseMachine.VECTOR]
        self.block_steps = block_steps
        self.block_num = block_num
        self.session = None
        self.step = 0

    def DrawGraph(self):
        self.train_input = tf.placeholder(tf.int32, shape=[1])
        self.train_label = tf.placeholder(tf.float32, shape=[1])
        self.state_input = tf.placeholder(tf.float32, shape=[1, self.embedding_size])
        embeddings = tf.constant(self.vectors)
        self.word_embed = tf.nn.embedding_lookup(embeddings, self.train_input)

        MergeMatrix = tf.Variable(tf.random_uniform([self.embedding_size * 3,
        self.embedding_size * 3], -1.0, 1.0))
        MergeBias = tf.Variable(tf.random_uniform([1, self.embedding_size * 3], -1.0, 1.0))
        layer1_out = tf.nn.sigmoid(tf.matmul(tf.concat(1 \
        , [self.state_input, self.word_embed, self.state_input * self.word_embed]) \
        , MergeMatrix) + MergeBias)

        self.weight = tf.Variable(tf.random_uniform([self.embedding_size * 3, 1], -1.0, 1.0))
        self.bias = tf.Variable(tf.zeros([1]))
        y = tf.nn.sigmoid(tf.matmul(layer1_out, self.weight) + self.bias)
        self.loss = tf.abs(self.train_label[0] - y[0, 0])
        self.optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(self.loss)
        # initialize saver
        self.saver = tf.train.Saver([MergeMatrix, MergeBias, self.weight \
        , self.bias])

    def Training(self):
        if self.session is None:
            self.session = tf.Session()
            self.session.run(tf.initialize_all_variables())
        average_loss = 0
        sentenceList = self.nextTrainSentences()
        if len(sentenceList) == 0:
            isEnd = True
        else:
            isEnd = False
        for sentence in sentenceList:
            state_input = np.zeros([1, self.embedding_size], np.float32)
            last = None
            for i in range(len(sentence)):
                if i < len(sentence) - 1:
                    label = 0.0
                else:
                    label = 1.0
                _, loss = self.session.run([
                    self.optimizer,
                    self.loss,
                ], feed_dict={
                    self.train_input: [sentence[i]],
                    self.train_label: [label],
                    self.state_input: state_input,
                })
                if math.isnan(loss):
                    assert not math.isnan(loss)
                average_loss += loss
        return average_loss, isEnd

    def TrainModel(self):
        while True:
            average_loss = 0
            isEnd = False
            for _ in range(self.block_num):
                loss, isEnd = self.Training()
                average_loss += loss
            if isEnd:
                break
            average_loss /= self.block_num
            print("Average loss at step ", self.step, ": ", average_loss)
            self.step += 1

    def SaveModel(self):
        self.saver.save(self.session, SentenceGramMachineV1.MODEL_FILENAME)
