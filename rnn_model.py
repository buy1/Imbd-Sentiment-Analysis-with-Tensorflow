import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt

class rnn_BASIC_model(object):
	"""
	A RNN for predicting output based on long term dependancies.
	"""
	def __init__(self, data_size, num_layers,stddev_init, batch_size, state_size, num_steps, vocab_size, num_classes,initial_state):
		self.batch_size = batch_size

	  
		cell_fn = tf.nn.rnn_cell.GRUCell
		
		self.x = tf.placeholder(tf.int32, shape=[batch_size, num_steps], name="input_x")
		self.y = tf.placeholder(tf.int32, shape=[batch_size], name='labels_placeholder')
		# self.initial=tf.placeholder(tf.int32, shape=[vocab_size,state_size])
		cell = cell_fn(state_size)
		self.cell = cell = tf.nn.rnn_cell.MultiRNNCell([cell] * num_layers)
	 


		#changes word to index and index to vector
		with tf.device("/cpu:0"):
			#randomize the initial variables of the matrix
			# a=input("a:")
			# initial_state=self.initial
			rnn_inputs = [tf.squeeze(i) for i in tf.split(tf.nn.embedding_lookup(initial_state, self.x),num_steps,1)]

		rnn_outputs, last_state = tf.contrib.rnn.static_rnn(cell, rnn_inputs, dtype=tf.float32)

   
		W_softmax = tf.Variable(tf.random_normal([state_size, num_classes]),name='W_softmax')
		b_softmax = tf.Variable(tf.random_normal([num_classes]),name='b_softmax')
		self.logits = tf.matmul(rnn_outputs[-1], W_softmax) + b_softmax 
		
	  
		self.losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y)
		self.loss = tf.reduce_mean(self.losses)

		# Accuracy
		self.probs = tf.nn.softmax(self.logits)
		self.predictions_label = tf.argmax(self.probs, 1, name="predictions")
		self.correct_predictions = tf.equal(self.predictions_label, tf.argmax(self.y,1))
		self.accuracy = tf.reduce_mean(tf.cast(self.correct_predictions, tf.float32), name="accuracy")
