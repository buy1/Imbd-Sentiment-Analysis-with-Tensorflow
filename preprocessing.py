import pandas as pd
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
import nltk.data
import logging
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import h5py
from gensim.models import Word2Vec
import os
from gensim.models.keyedvectors import KeyedVectors
import pickle
# Read the data from files

#All the data has variable length so we try averaging the vectors for each review
#word is the review


def review_to_wordlist(review, remove_stopwords=False):
	#Function to convert document to a sequence of words and optionally remove stopword
	#removes html elements and alphabet words
	review_text=re.sub("[^a-zA-Z]"," ",BeautifulSoup(review,"html.parser").get_text())
	#puts the words in lowercase and tokenizes it by whitespace
	words= review_text.lower().split()
	#optionally if remove_stopwords is true
	if remove_stopwords:
		#changes the array into a set
		stops=set(stopwords.words("english"))
		#removes the stopwords
		words=[w for w in words if not w in stops]
	return (words)
#
def makeFeatureVec(wordlist, model, num_features,index2word_set):
	#pre-initialize for speed
	# featureVec=np.zeros((len(wordlist),num_features),dtype="float32")
	featureVec=[]
	for word in wordlist:
		if word in index2word_set:
			featureVec.append(model[word])
	return featureVec

#TODO: Make this for a word by a word
def getAvgFeatureVecs(reviews, model, num_features):
	reviewFeatureVecs = np.zeros((len(reviews),num_features),dtype="float32")
	counter=0
	index2word_set=set(model.index2word)
	for review in reviews:
		np.add(reviewFeatureVecs[counter],makeFeatureVec(review,model,num_features,index2word_set))
		counter+=1
	return reviewFeatureVecs

model=KeyedVectors.load_word2vec_format("glove-word2vec.bin", binary=True)

labeledpath=os.getcwd()+"/data/labeledTrainData_shuffled.tsv"
testpath=os.getcwd()+"/data/testData_shuffled.tsv"
unlabeled_path=os.getcwd()+"/data/unlabeledTrainData_shuffled.tsv"

traindata=pd.read_csv(labeledpath, header=0,delimiter="\t", quoting=3)
local_testdata=traindata[22500:25000] 	
traindata1=traindata[0:22500]


testdata=pd.read_csv(testpath,header=0,delimiter="\t",quoting=3)
unlabeled_traindata=pd.read_csv(unlabeled_path,header=0,delimiter="\t", quoting=3)		
print ("There is this much traindata: " + str(traindata["review"].size))
print ("There is this much testdata: " + str(testdata["review"].size))
print ("There is this much unlabeled data: " + str(unlabeled_traindata["review"].size))

xtrain=[]
print ("converting review to word list for traindata")
for review in traindata1["review"]:
	xtrain.append(review_to_wordlist(review,remove_stopwords=True))	
ytrain=list(traindata1["sentiment"])

print ("converting local test reviews to word list")
local_xtest=[]
for review in local_testdata["review"]:
	local_xtest.append(review_to_wordlist(review,remove_stopwords=True))
local_ytest=list(local_testdata["sentiment"])

print ("converting review to word list for testdata")
xtest=[]
for review in testdata["review"]:
	xtest.append(review_to_wordlist(review,remove_stopwords=True))

print ("converting review to word list for unlabeled train data")
x_unlabeled=[]
for review in unlabeled_traindata:
	x_unlabeled.append(review_to_wordlist(review,remove_stopwords=True))

print ("Making Word2Vec vector")
num_features=300
dataVecs_xtrain=getAvgFeatureVecs(xtrain,model,num_features)
dataVecs_xtest=getAvgFeatureVecs(xtest,model,num_features)
dataVecs_x_unlabeled=getAvgFeatureVecs(x_unlabeled,model,num_features)

print ("building vocab for xtrain")
#makes a list of all unique vocab words
xtrain_vocab=[]
for review in traindata["review"]:
	xtrain_vocab.append(review_to_wordlist(review))
index2word=[]
for review in xtrain_vocab:
	for word in review:
		index2word.append(word)
print ("and xtest")
for review in xtest:
	for word in review:
		index2word.append(word)
print ("and x_unlabeled")
for review in x_unlabeled:
	for word in review:
		index2word.append(word)
#removes all non-unique elements
index2word=list(set(index2word))
#makes a dictionary with the words' value being their index
vocab={}
for idx,word in enumerate(index2word):
	vocab[word]=idx

print (len(vocab))
print ("word 2 index for xtrain")
wordvector_xtrain=[]
for review in xtrain:
	indexlist=[]
	for word in review:
		indexlist.append(vocab[word])
	wordvector_xtrain.append(indexlist)

print ("word 2 index for xtest")
wordvector_xtest=[]
for review in xtest:
	indexlist=[]
	for word in review:
		indexlist.append(vocab[word])
	wordvector_xtest.append(indexlist)

print ("word 2 index for local xtest")
local_wordvector_xtest=[]
for review in local_xtest:
	indexlist=[]
	for word in review:
		indexlist.append(vocab[word])
	local_wordvector_xtest.append(indexlist)

index2word_set=set(model.index2word)

print ("Saving data...")

# Word2Vec Vector
f=h5py.File("wordVecs.hdf5","w")

dset=f.create_dataset("xtest",data=dataVecs_xtest,dtype="float32")
dset1=f.create_dataset("xtrain",data=dataVecs_xtrain,dtype="float32")
dset2=f.create_dataset("x_unlabeled",data=dataVecs_x_unlabeled,dtype="float32")
f.close()


# print ("Saving dataVecs for xtest")
# # np.save("xtest",np.array(dataVecs_xtest))
# print ("Saving dataVecs for xtrain")
# np.save("xtrain",np.array(dataVecs_xtrain))
# print ("Saving dataVecs for x_unlabeled")
# np.save("x_unlabeled",np.array(dataVecs_x_unlabeled))

#local test
print ("Saving local xtest wordvector")
np.save("local_wordvector_xtest",np.array(local_wordvector_xtest))
print ("Saving local_ytest")
np.save("local_ytest",np.array(local_ytest))

#training index vector and data
print ("Saving training data")
np.save("wordvector_xtrain.npy",np.array(wordvector_xtrain))
np.save("xtrain.npy",np.array(xtrain))
np.save("ytrain.npy",np.array(ytrain))

#kaggle testing data
print ("Saving testing data")
np.save("wordvector_xtest.npy",np.array(wordvector_xtest))
np.save("xtest.npy",np.array(xtest))
