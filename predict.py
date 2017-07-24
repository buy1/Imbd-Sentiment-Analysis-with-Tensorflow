import tensorflow as tf
import numpy as np
import pandas as pd
from rnn_model import *
import os
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



# x_test=np.load("./wordvector_xtest.npy")

x_test=np.load("./local_wordvector_xtest.npy")
y_test=np.load("./local_ytest.npy")

# x_test=np.load("./local_xtest_word2index.npy")
# y_test=np.load("./local_ytest_word2index.npy")

config=tf.ConfigProto(log_device_placement=False)
with tf.Graph().as_default():
	with tf.Session(config=config) as sess:
		batch_size=x_test.shape[0]
		state1_size=100
		MAXLEN = 30
		num_classes=2
		sqrt_0 = math.sqrt(1.0 / float(state1_size))
		sqrt_1 = math.sqrt(1.0 / float(state1_size))
		num_layers=1
		data_size=x_test.shape[0]
		vocab_size =101399
		#use dropout or not
		rnn = rnn_BASIC_model(batch_size =batch_size,
						state_size =state1_size,
						num_steps = MAXLEN,
						num_classes = num_classes,
						stddev_init = [sqrt_0, sqrt_1],
						num_layers = num_layers,
						data_size = data_size,
						vocab_size = vocab_size)
	   
		sess.run(tf.global_variables_initializer())
		# print (rnn.predictions_label.shape)
		# print (x_train.shape)
		saver = tf.train.Saver(tf.global_variables())
		saver.restore(sess,"LSTM_RNN_MODEL.ckpt")
		x=prepare_data(x_test,MAXLEN)
		prediction_label= sess.run([rnn.predictions_label],
									{rnn.x: x})
correct_answers=0
prediction_label=prediction_label[0]

for i in range(len(prediction_label)):
	if prediction_label[i]==y_test[i]:
		correct_answers+=1
# correct_answers = (prediction_label) == np.array(y_test).sum()
print ("Accuracy: "+ str((correct_answers/batch_size)*100)+"%")
# prediction_label=prediction_label[0]
# testpath=os.getcwd()+"/data/testData.tsv"
# testdata=pd.read_csv(testpath,header=0,delimiter="\t",quoting=3)

# output = pd.DataFrame( data={"id":testdata["id"], "sentiment":prediction_label} )
# output.to_csv( "LSTM_RNN.csv", index=False, quoting=3 )

