import csv
import random
import os

def shuffle(path,newpath):
	lines = open(path).readlines()
	firstline=[lines[0]]
	restoflines=lines[1:]
	# print (type(firstline))
	# print (type(restoflines))
	random.shuffle(restoflines)
	firstline.extend(restoflines)

	open(os.getcwd()+"/data/"+newpath+"_shuffled.tsv", 'w').writelines(firstline)

labeledpath=os.getcwd()+"/data/labeledTrainData.tsv"
testpath=os.getcwd()+"/data/testData.tsv"
unlabeled_path=os.getcwd()+"/data/unlabeledTrainData.tsv"

shuffle(labeledpath,"labeledTrainData")
shuffle(testpath,"testData")
shuffle(unlabeled_path,"unlabeledTrainData")
