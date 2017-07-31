# Imbd-Sentiment-Analysis-with-Tensorflow
Made in python 3.6

Requires glove-word2vec.bin,tensorflow,numpy,sklearn, h5py

Run preprocessing.py, rnn_training.py, predict.py in that order.

Model is a Recurrent Neural Network with GRU cells with the word vectors initialized using word2vec.

Highest accuracy achieved is 89% with a batch size of 50, 9 epochs, 100 hidden states, .01 learning rate,max sentence length of 200, and a single layer.
