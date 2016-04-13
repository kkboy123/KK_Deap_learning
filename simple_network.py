__author__ = 'kkboy'


# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 15:10:35 2015
@author: joao
"""

import tensorflow as tf
import cPickle
import sys
import numpy as np
from pdb import set_trace as pdb
import matplotlib.pylab as plt

VALIDATION_SIZE = 5000  # Size of the validation set.
SEED = 66478  # Set to None for random seed.
BATCH_SIZE = 64
NUM_EPOCHS = 10
HIDDEN_NODES = 40

def error_rate(predictions, labels):
  """Return the error rate based on dense predictions and 1-hot labels."""
  return np.mean(np.sqrt(((predictions-labels))**2))*100

#get data
f = file('nn_val.dat','rb')
val_data   = cPickle.load(f)
val_labels = cPickle.load(f)
f.close()

f = file('nn_train.dat','rb')
train_data   = cPickle.load(f)
train_labels = cPickle.load(f)
f.close()

norm = np.std(train_labels)
val_data[:,:15] /=norm
val_labels /= norm
train_data[:,:15] /=norm
train_labels /= norm
norm = np.std(train_data[:,15:],axis=0)
train_data[:,15:] /= norm
val_data[:,15:] /= norm

N_FEATURES = val_data.shape[1]
train_size = train_labels.shape[0]
# This is where training samples and labels are fed to the graph.
# These placeholder nodes will be fed a batch of training data at each
# training step using the {feed_dict} argument to the Run() call below.
train_data_node = tf.placeholder(
            tf.float32,
            shape=(BATCH_SIZE,N_FEATURES))

train_labels_node = tf.placeholder(tf.float32,
            shape=(BATCH_SIZE,1))

# For the validation and test data, we'll just hold the entire dataset in
# one constant node.
validation_data_node = tf.constant(val_data)

fc1_weights = tf.Variable(  # fully connected
                tf.truncated_normal(
                    [N_FEATURES, HIDDEN_NODES],
                    stddev=0.1,
                    seed=SEED))
fc1_biases = tf.Variable(tf.constant(0.1, shape=[HIDDEN_NODES]))

fc2_weights = tf.Variable(  # fully connected
                tf.truncated_normal(
                    [HIDDEN_NODES, 1],
                    stddev=0.1,
                    seed=SEED))
fc2_biases = tf.Variable(tf.constant(0.1, shape=[1]))


def model(data, train=False):
    hidden = tf.nn.relu(tf.matmul(data, fc1_weights) + fc1_biases)
    return tf.matmul(hidden, fc2_weights) + fc2_biases

train_prediction = model(train_data_node, True)
loss = tf.reduce_mean(tf.sqrt(tf.pow(((tf.sub(train_prediction,train_labels_node))),2)),name='RMS')


batch_index = tf.Variable(0)
## Decay once per epoch, using an exponential schedule starting at 0.01.
learning_rate = tf.train.exponential_decay(
      1E-8,                # Base learning rate.
      batch_index * BATCH_SIZE,  # Current index into the dataset.
      train_size,          # Decay step.
      0.95,                # Decay rate.
      staircase=True)


optimizer = tf.train.MomentumOptimizer(learning_rate,
                                         0.9).minimize(loss,
                                                       global_step=batch_index)
# We'll compute them only once in a while by calling their {eval()} method.
validation_prediction = model(validation_data_node)
train_error_list = []
val_error_list = []
with tf.Session() as s:
    tf.initialize_all_variables().run()
    print('Initialized!')
    for step in xrange(NUM_EPOCHS * train_size // BATCH_SIZE):
        # Compute the offset of the current minibatch in the data.
        # Note that we could use better randomization across epochs.
        offset = (step * BATCH_SIZE) % (train_size - BATCH_SIZE)
        batch_data = train_data[offset:(offset + BATCH_SIZE), :]
        batch_labels = train_labels[offset:(offset + BATCH_SIZE)]
        # This dictionary maps the batch data (as a numpy array) to the
        # node in the graph is should be fed to.
        feed_dict = {train_data_node: batch_data,
                     train_labels_node: batch_labels}
        # Run the graph and fetch some of the nodes.
        _,l, lr, predictions,f1_weights,f2_weights = s.run(
            [optimizer,loss, learning_rate, train_prediction,fc1_weights,fc2_weights],
            feed_dict=feed_dict)
        if step % 100 == 0:
            print np.max(f1_weights),np.min(f1_weights)
            print np.max(f2_weights),np.min(f2_weights)
            print('Epoch %.2f' % (float(step) * BATCH_SIZE / train_size))
            print('Minibatch loss: %.3f, learning rate: %.6f' % (l, lr))
            train_error = error_rate(predictions, batch_labels)
            val_error = error_rate(validation_prediction.eval(),val_labels)
            print('Minibatch error: %.1f%%' % train_error)
            print('Validation error: %.1f%%' % val_error)
            train_error_list.append(train_error)
            val_error_list.append(val_error)
            sys.stdout.flush()
