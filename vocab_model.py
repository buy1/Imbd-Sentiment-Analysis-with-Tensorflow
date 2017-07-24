import pandas as pd
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
import nltk.data
import logging
from gensim.models import word2vec
# Read the data from files
	
#Takes in a raw_String and removes html elements and non alphabetical characters
#returns a list of words
def review_to_wordlist(review, remove_stopwords=False):
	#Function to convert document to a sequence of words and optionally remove stopwords

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

#tokenizes the review and setences and optionally removes the stopwords
def sent_tokenizer(review,remove_stopwords=False):
	#changes a review to a list of sentences and each sentence is a a list of words
	sentences=nltk.sent_tokenize(review)
	#removes lists that contain empty strings empty tuples ,zeros
	collapsed_sent=[x for x in sentences if x]
	tokenized_sent=[]
	for sent in collapsed_sent:
		#calls review_to_wordlist on the sentence and splits it by whitespace
		wordlist=review_to_wordlist(sent,remove_stopwords)
		for word in wordlist:
			tokenized_sent.append(word)
	return tokenized_sent

if __name__ == '__main__':
	labeledpath="/Users/bhuang/Desktop/bagsof(words and popcorns)/data/labeledTrainData.tsv"
	testpath="/Users/bhuang/Desktop/bagsof(words and popcorns)/data/testData.tsv"
	unlabeled_path="/Users/bhuang/Desktop/bagsof(words and popcorns)/data/unlabeledTrainData.tsv"

	traindata=pd.read_csv(labeledpath, header=0,delimiter="\t", quoting=3)
	testdata=pd.read_csv(testpath,header=0,delimiter="\t",quoting=3)
	unlabeled_traindata=pd.read_csv(unlabeled_path,header=0,delimiter="\t", quoting=3)
	print ("There is this much traindata: " + str(traindata["review"].size))
	print ("There is this much testdata: " + str(testdata["review"].size))
	print ("There is this much unlabeled traindata" + str(unlabeled_traindata["review"].size))

	#adds both the labeled and unlabeled set to sentences and passes it to the word2vec classifier
	sentences=[]
	print ("Parsing sentences from training set")
	for review in traindata["review"]:
		sentences.append(sent_tokenizer(review))

	print ("Parsing sentences from unlabeled set")
	for review in unlabeled_traindata["review"]:
		sentences.append(sent_tokenizer(review))

	print ("Parsing sentences from test set")
	for review in testdata["review"]:
		sentences.append(sent_tokenizer(review))


	logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',level=logging.INFO)

	num_features=300 #word vector dimensionality
	min_word_count=40 #min word count
	num_workers=40 # number of threads run in parallel
	context=10 #context window size
	downsampling=1e-3 # downsample setting for frequent words

	print ("Training model...")
	model = word2vec.Word2Vec(sentences, workers=num_workers, \
	            size=num_features, min_count = min_word_count, \
	            window = context, sample = downsampling)


	model.init_sims(replace=True)
	model=model.wv
	model_name = "word2vec_model.bin"
	model.save(model_name)







	# 