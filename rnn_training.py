import tensorflow as tf
import numpy as np
import datetime
from rnn_model import rnn_BASIC_model
from tensorflow.python.framework import dtypes
import tensorflow as tf
import numpy as np
import os
import time
import datetime
from tensorflow.contrib import learn
from sklearn.metrics import confusion_matrix
from six.moves import cPickle
import math
import time
import os
import matplotlib.pyplot as plt
import h5py
def prepare_data(seqs, maxlen=None):

    lengths = [len(s) for s in seqs]

    if maxlen is not None:
        new_seqs = []
        new_lengths = []
        for l, s in zip(lengths, seqs):
            new_seqs.append(s)
            new_lengths.append(l)
        lengths = new_lengths
        seqs = new_seqs
        if len(lengths) < 1:
            return None, None
    n_samples = len(seqs)
    x = np.zeros((maxlen, n_samples))
    for idx, s in enumerate(seqs):
        x[:lengths[idx], idx] = np.array([s[:maxlen]])
    return x.T

def prepare_data(seqs, labels, maxlen=None):
    """Create the matrices from the datasets.
    This pad each sequence to the same lenght: the lenght of the
    longuest sequence or maxlen.
    if maxlen is set, we will cut all sequence to this maximum
    lenght.
    This swap the axis!
    """
    # x: a list of sentences
    lengths = [len(s) for s in seqs]

    if maxlen is not None:                                                                                                                                                                                                                                                                                                                                                                                              
        new_seqs = []
        new_labels = []
        new_lengths = []
        for l, s, y in zip(lengths, seqs, labels):
            # if l < maxlen:
            new_seqs.append(s)
            new_labels.append(y)
            new_lengths.append(l)
        lengths = new_lengths
        labels = new_labels
        seqs = new_seqs

        if len(lengths) < 1:
            return None, None, None

    n_samples = len(seqs)
    # if len(lengths) > 0:
    #     maxlen = numpy.max(lengths)

    x = np.zeros((maxlen, n_samples))
    # x_mask = np.zeros((maxlen, n_samples))#.astype(theano.config.floatX)
    for idx, s in enumerate(seqs):
        # print (x[:lengths[idx], idx].shape)
        # print (np.asmatrix(s[:maxlen]).T.shape)
        # print (x[:lengths[idx], idx].shape)
        # print (np.array([s[:maxlen]]).shape)
        #this is the main padding part where if length 
        x[:lengths[idx], idx] = np.array([s[:maxlen]])
        # x_mask[:lengths[idx], idx] = 1.

    return x.T, np.max(lengths), labels



state1_size=100
# state2_size=100
# state3_size=100
# keep_prob_layer1=.9
# keep_prob_layer2=.8
# keep_prob_layer3=.7

# Training parameters'
batch_size=50
#number of training steps
nb_epochs=30
#number of steps between every eval print
eval_every=100
#inital learning rate
learning_rate=.01
# l2_reg_lambda=0.001
num_steps=20
num_classes=2
num_layers=1

MAXLEN = 200
vocab_size = 101399
#use dropout or not
is_training=True
save_dir="~/Desktop/"

#takes the pre trained data
# seqs=np.load("./x_train.npy")
labels=np.load("./ytrain.npy")
seqs=np.load("./wordvector_xtrain.npy")
model=h5py.File('wordVecs.hdf5','r')
dataVecs_xtest=model['xtest'][:]
dataVecs_xtrain=model['xtrain'][:]
dataVecs_x_unlabeled=model['x_unlabeled'][:]


xtrain=dataVecs_xtrain[:]
xtrain=xtrain[:,:100]
pad=(0,vocab_size-xtrain.shape[0])
xtrain=np.pad(xtrain,((0,vocab_size-xtrain.shape[0]),(0,0)),'constant',constant_values=(0))

print(xtrain.shape)
print (type(xtrain))

data_size=seqs.shape[0]


config=tf.ConfigProto(log_device_placement=False)
with tf.Graph().as_default():
    with tf.Session(config=config) as sess:

        sqrt_0 = math.sqrt(1.0 / float(state1_size))
        sqrt_1 = math.sqrt(1.0 / float(state1_size))
        # sqrt_2 = math.sqrt(1.0 / float(state1_size))
        # sqrt_3 = math.sqrt(1.0 / float(state1_size))
        # list_stddev = [sqrt_0, sqrt_1, sqrt_2, sqrt_3]
        # current_list = [sqrt for sqrt in list_stddev[:num_layers]]

        initial_state = tf.Variable(xtrain)
        rnn = rnn_BASIC_model(batch_size =batch_size,
                        state_size =state1_size,
                        num_steps = MAXLEN,
                        num_classes = 2,
                        stddev_init = [sqrt_0, sqrt_1],
                        num_layers =num_layers,
                        data_size = data_size,
                        vocab_size = vocab_size,
                        initial_state = initial_state)


        training_losses = []

        global_step = tf.Variable(0)
        init_lr =learning_rate

        optimizer = tf.train.AdagradOptimizer(init_lr)
        train_step = optimizer.minimize(rnn.loss)

        tf.global_variables_initializer().run()

        saver = tf.train.Saver(tf.global_variables())

        count=0
        start_t = time.time()
        for epoch in range(nb_epochs):
            print ("epoch %d" % epoch)
            epoch_size = data_size // rnn.batch_size
            start_time = time.time()
            costs = 0.0
            correct_answers = 0.0
            for step in range(epoch_size):
                x = seqs[step*rnn.batch_size:(step+1)*rnn.batch_size]
                y = labels[step*rnn.batch_size:(step+1)*rnn.batch_size]
                x, max_len_seqs, y = prepare_data(x, y, MAXLEN)
                if x is not None:
                    x = x[:,:MAXLEN]
                    cost, prediction, _ = sess.run([rnn.loss, rnn.probs, train_step],
                                                 {rnn.x: x,
                                                  rnn.y: y})
                    correct_answers += (np.argmax(prediction, 1) == np.array(y)).sum()
                    costs += cost

                    if step % 300 == 0 and step > 0 :
                        print ("At step %d - Loss : %.3f  - Accuracy : %.3f " % (step, costs / step, correct_answers / (step * rnn.batch_size)))
            savePath=saver.save(sess,'LSTM_RNN_MODEL_'+str(count)+'.ckpt')
            count+=1
            #shuffles the data
            arr=list(zip(labels,seqs))
            np.random.shuffle(arr)
            labels,seqs=zip(*arr)

