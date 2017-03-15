import numpy as np
import random
import collections
import tensorflow as tf
from six.moves import xrange
import math
from WordVector.BaseVectorTrainer import *

class ChineseVectorTrainer(BaseVectorTrainer):
    def generate_batch(self, data_index, batch_size, num_skips, skip_window):
        assert batch_size % num_skips == 0
        assert num_skips <= 2 * skip_window
        batch = np.ndarray(shape=(batch_size), dtype=np.int32)
        labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
        span = 2 * skip_window + 1
        buf = collections.deque(maxlen=span)
        for _ in range(span):
            buf.append(self.data[data_index])
            data_index = (data_index + 1) % len(self.data)
        for i in range(batch_size // num_skips):
            target = skip_window
            targets_to_avoid = [skip_window]
            for j in range(num_skips):
                while target in targets_to_avoid:
                    target = random.randint(0, span - 1)
                targets_to_avoid.append(target)
                if buf[skip_window] in self.dictionary:
                    batch[i * num_skips + j] = self.dictionary[buf[skip_window]]["index"]
                else:
                    batch[i * num_skips + j] = 0
                if buf[target] in self.dictionary:
                    labels[i * num_skips + j, 0] = self.dictionary[buf[target]]["index"]
                else:
                    labels[i * num_skips + j, 0] = 0
            buf.append(self.data[data_index])
            data_index = (data_index + 1) % len(self.data)
        return batch, labels

    def ShowData(self, data_index, batch_size, num_skips, skip_window):
        batch, labels = self.generate_batch(data_index, batch_size, num_skips, skip_window)
        for i in range(batch_size):
            print(batch[i], self.reverse_dictionary[batch[i]], \
              "->", labels[i, 0], self.reverse_dictionary[labels[i, 0]])

    def InitConfig(self, data_index=0, batch_size=128, embedding_size=128, \
    skip_window=1, num_skips=2, valid_size=16, valid_window=100, num_sampled=64, \
    num_steps=10000):
        self._data_index = data_index
        self._batch_size = batch_size
        self._embedding_size = embedding_size
        self._skip_window = skip_window
        self._num_skips = num_skips
        self._valid_size = valid_size
        self._valid_window = valid_window
        self._num_sampled = num_sampled
        self._num_steps = num_steps

    def TrainModal(self):
        graph, train_inputs, train_labels, optimizer, loss, normalized_embeddings = self._drawGraph()
        self._trainData(graph, train_inputs, train_labels, optimizer, loss, normalized_embeddings)

    def _drawGraph(self):
        batch_size = self._batch_size
        # Dimension of the embedding vector.
        embedding_size = self._embedding_size
        # Random set of words to evaluate similarity on.
        valid_size = self._valid_size
        # Only pick dev samples in the head of the distribution.
        valid_window = self._valid_window
        valid_examples = np.random.choice(valid_window, valid_size, replace=False)
        # Number of negative examples to sample.
        num_sampled = self._num_sampled

        graph = tf.Graph()
        with graph.as_default():
            # Input data
            train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
            train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
            valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

            embeddings = tf.Variable(tf.random_uniform([self.length, \
              embedding_size], -1.0, 1.0))
            embed = tf.nn.embedding_lookup(embeddings, train_inputs)
            # Construct the variables for the NCE loss
            nce_weights = tf.Variable( \
              tf.truncated_normal([self.length, embedding_size], \
              stddev=1.0 / math.sqrt(embedding_size)), name="nce_weights")
            nce_biases = tf.Variable(tf.zeros([self.length]), name="nce_biases")
            loss = tf.reduce_mean( \
              tf.nn.nce_loss(nce_weights, nce_biases, embed, train_labels, \
                num_sampled, self.length))
            # Construct the SGD optimizer using a learning rate of 1.0.
            optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)
            # Compute the cosine similarity between minibatch examples and all embeddings.
            norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
            normalized_embeddings = embeddings / norm
            valid_embeddings = tf.nn.embedding_lookup( \
                normalized_embeddings, valid_dataset)
            similarity = tf.matmul(valid_embeddings, \
                normalized_embeddings, transpose_b=True)

            self.tf_nce_weights = nce_weights
            self.tf_nce_biases = nce_biases
        return graph, train_inputs, train_labels, optimizer, loss, normalized_embeddings

    def _trainData(self, graph, train_inputs, train_labels, optimizer, loss, normalized_embeddings):
        data_index = self._data_index
        batch_size = self._batch_size
        # Step 5: Begin training.
        num_steps = self._num_steps
        # How many times to reuse an input to generate a label.
        num_skips = self._num_skips
        # How many words to consider left and right.
        skip_window = self._skip_window
        # Random set of words to evaluate similarity on.
        valid_size = self._valid_size
        # Only pick dev samples in the head of the distribution.
        valid_window = self._valid_window
        valid_examples = np.random.choice(valid_window, valid_size, replace=False)

        with tf.Session(graph=graph) as session:
            # We must initialize all variables before we use them.
            tf.initialize_all_variables().run()

            average_loss = 0
            for step in xrange(num_steps):
                batch_inputs, batch_labels = self.generate_batch( \
                    data_index, batch_size, num_skips, skip_window)
                data_index += 1
                feed_dict = {train_inputs : batch_inputs, train_labels : batch_labels}

                # We perform one update step by evaluating the optimizer op (including it
                # in the list of returned values for session.run()
                _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
                average_loss += loss_val

                if step % 2000 == 0:
                    if step > 0:
                        average_loss /= 2000
                        # The average loss is an estimate of the loss over the last 2000 batches.
                        print("Average loss at step ", step, ": ", average_loss)
                        average_loss = 0

                # Note that this is expensive (~20% slowdown if computed every 500 steps)
                if step % 10000 == 0 and step > 0:
                    sim = similarity.eval()
                    for i in xrange(valid_size):
                        valid_word = self.reverse_dictionary[valid_examples[i]]["word"]
                        top_k = 8 # number of nearest neighbors
                        nearest = (-sim[i, :]).argsort()[1:top_k+1]
                        log_str = "Nearest to %s:" % valid_word
                        for k in xrange(top_k):
                            close_word = self.reverse_dictionary[nearest[k]]["word"]
                            log_str = "%s %s," % (log_str, close_word)
                        print(log_str)
            final_embeddings = normalized_embeddings.eval()

            # save weight bias
            self.SaveModal(session)

    def SaveModal(self, session):
        self.weights = self.tf_nce_weights.eval()
        self.biases = self.tf_nce_biases.eval()
