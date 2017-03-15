from WordVector.BaseVectorTrainer import BaseVectorMachine
from langPreprocesser.ChinesePreprocesser import Article, Sentence
import tensorflow as tf
from six.moves import xrange
import numpy as np
import Util
import math

class ChineseVectorMachineV2(BaseVectorMachine):
    VECTOR = "vector"
    NCE_WEIGHT = "nce_weight"
    NCE_BIAS = "nce_bias"

    def __init__(self):
        BaseVectorMachine.__init__(self)
        self.remainBatchList = []
        self.remainLabelsList = []
        self.session = None

    def GenerateBatch(self, batch_size, skip_window):
        batchList = self.remainBatchList
        labelsList = self.remainLabelsList
        cnt = len(self.remainBatchList)
        while cnt < batch_size:
            sentence = self._nextSentence()
            if sentence is None:
                break
            wordList = []
            # filter symbols
            for word in sentence.words:
                if Util.HasWord(word):
                    wordList.append(word)
            for i in xrange(len(wordList)):
                train_word = wordList[i]
                if train_word in self.dictionary:
                    train_word_index = self.dictionary[train_word]["index"]
                else:
                    train_word_index = 0
                for j in range(i - skip_window, i + skip_window + 1):
                    if (j < 0) or (j >= len(wordList)) or j == i:
                        continue
                    target_word = wordList[j]
                    # append index
                    batchList.append(train_word_index)
                    if target_word in self.dictionary:
                        labelsList.append(self.dictionary[target_word]["index"])
                    else:
                        labelsList.append(0)
                    cnt += 1
        if cnt >= batch_size:
            self.remainBatchList = batchList[batch_size:]
            self.remainLabelsList = labelsList[batch_size:]
        elif cnt < batch_size:
            self.remainBatchList = batchList
            self.remainLabelsList = labelsList
            batchList = []
            labelsList = []
        return np.array(batchList[0:batch_size], dtype=np.int32), \
        np.array(labelsList[0:batch_size], dtype=np.int32, ndmin=2).T

    def InitConfig(self, batch_size=128, embedding_size=128, skip_window=1, \
    num_sampled=64, block_steps=-1, iscontinue=False, num_block=1000):
        self.batch_size = batch_size
        self.embedding_size = embedding_size
        self.skip_window = skip_window
        self.num_sampled = num_sampled
        self.block_steps = block_steps
        self.num_block = num_block
        self.vectors = None
        self.nce_weights_vectors = None
        self.nce_biases_vectors = None
        if iscontinue and (ChineseVectorMachineV2.VECTOR in self.reverse_dictionary[1]) and \
        (ChineseVectorMachineV2.NCE_WEIGHT in self.reverse_dictionary[1]) and \
        (ChineseVectorMachineV2.NCE_BIAS in self.reverse_dictionary[1]):
            self.RestoreState()

    def RestoreState(self):
        if ChineseVectorMachineV2.VECTOR not in self.reverse_dictionary[1] or \
        len(self.reverse_dictionary[1][ChineseVectorMachineV2.VECTOR]) != self.embedding_size:
            print("the embedding size is conflict.")
            return
        # if ChineseVectorMachineV2.NCE_WEIGHT not in self.reverse_dictionary[1] or \
        # len(self.reverse_dictionary[1][ChineseVectorMachineV2.NCE_WEIGHT]) != self.embedding_size:
        #     print("the nce weight size is conflict.")
        #     return
        # if ChineseVectorMachineV2.NCE_BIAS not in self.reverse_dictionary[1]:
        #     print("the nce bias is None.")
        #     return
        self.vectors = np.zeros([self.length + 1, self.embedding_size], np.float32)
        self.vectors[0, :] = np.random.random(self.embedding_size)
        # self.nce_weights_vectors = np.zeros([self.length + 1, self.embedding_size], np.float32)
        # self.nce_weights_vectors[0, :] = np.random.random(self.embedding_size)
        # self.nce_biases_vectors = np.zeros([self.length + 1], np.float32)
        for i in xrange(self.length):
            self.vectors[i + 1, :] = np.array(self.reverse_dictionary[i + 1][ChineseVectorMachineV2.VECTOR], \
            np.float32)
            # self.nce_weights_vectors[i + 1, :] = np.array(self.reverse_dictionary[i + 1][ChineseVectorMachineV2.NCE_WEIGHT], \
            # np.float32)
            # self.nce_biases_vectors[i + 1] = np.array(self.reverse_dictionary[i + 1][ChineseVectorMachineV2.NCE_BIAS], \
            # np.float32)

    def DrawGraph(self):
        # Input data
        train_inputs = tf.placeholder(tf.int32, shape=[self.batch_size])
        train_labels = tf.placeholder(tf.int32, shape=[self.batch_size, 1])

        if self.vectors is None:
            embeddings = tf.Variable(tf.random_uniform([self.length + 1, \
            self.embedding_size], -1.0, 1.0))
        else:
            embeddings = tf.Variable(self.vectors)
        embed = tf.nn.embedding_lookup(embeddings, train_inputs)
        # Construct the variables for the NCE loss
        if self.nce_weights_vectors is None:
            nce_weights = tf.Variable( \
            tf.truncated_normal([self.length + 1, self.embedding_size], \
            stddev=1.0 / math.sqrt(self.embedding_size)), name="nce_weights")
        else:
            nce_weights = tf.Variable(self.nce_weights_vectors, name="nce_weights")
        if self.nce_biases_vectors is None:
            nce_biases = tf.Variable(tf.zeros([self.length + 1]), name="nce_biases")
        else:
            nce_biases = tf.Variable(self.nce_biases_vectors, name="nce_biases")
        loss = tf.reduce_mean( \
        tf.nn.nce_loss(nce_weights, nce_biases, embed, train_labels, \
        self.num_sampled, self.length + 1))
        # Construct the SGD optimizer using a learning rate of 1.0.
        optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

        return train_inputs, train_labels, loss, optimizer, embeddings, \
        nce_weights, nce_biases

    def Running(self, isTest, train_inputs, train_labels, loss, optimizer, embeddings):
        if self.session is None:
            self.session = tf.Session()
            # We must initialize all variables before we use them.
            self.session.run(tf.initialize_all_variables())
            self.step = 0
        with self.session.as_default():
            while (self.block_steps < 0) or (self.step < self.block_steps):
                average_loss = 0
                isEnd = False
                for _ in xrange(self.num_block):
                    batch_inputs, batch_labels = self.GenerateBatch( \
                    self.batch_size, self.skip_window)
                    if len(batch_inputs) < self.batch_size:
                        isEnd = True
                        break
                    feed_dict = {train_inputs : batch_inputs, train_labels : batch_labels}

                    # We perform one update step by evaluating the optimizer op (including it
                    # in the list of returned values for session.run()
                    if isTest:
                        loss_val = self.session.run(loss, feed_dict=feed_dict)
                    else:
                        _, loss_val = self.session.run([optimizer, loss], feed_dict=feed_dict)
                    average_loss += loss_val

                if isEnd:
                    break
                average_loss /= self.num_block
                # The average loss is an estimate of the loss over the last 2000 batches.
                print("Average loss at step ", self.step, ": ", average_loss)
                self.step += 1
            self.SaveModal()

    def SaveModal(self):
        self.vectors = self.embeddings.eval()
        # self.nce_weights_vectors = self.session.run(self.nce_weights.eval())
        # self.nce_biases_vectors = self.session.run(self.nce_biases.eval())

    def StartDrawing(self):
        self.train_inputs, self.train_labels, self.loss, \
        self.optimizer, self.embeddings, self.nce_weights, \
        self.nce_biases = self.DrawGraph()

    def StartRunning(self, isTest=False):
        self.Running(isTest, self.train_inputs, self.train_labels, \
        self.loss, self.optimizer, self.embeddings)

    def GetResultWordList(self):
        wordList = []
        for index in xrange(self.length):
            wordItem = self.reverse_dictionary[index + 1]
            wordItem[ChineseVectorMachineV2.VECTOR] = self.vectors[index, :].tolist()
            # wordItem[ChineseVectorMachineV2.NCE_WEIGHT] = self.nce_weights_vectors[index, :].tolist()
            # wordItem[ChineseVectorMachineV2.NCE_BIAS] = self.nce_biases_vectors[index].tolist()
            del wordItem["_id"]
            wordList.append(wordItem)
        return wordList

    def SaveState(self):
        wordsList = self.GetResultWordList()
        for wordItem in wordsList:
            Util.UpdateWord(wordItem)
