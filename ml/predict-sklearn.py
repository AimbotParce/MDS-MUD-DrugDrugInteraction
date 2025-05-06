#!/usr/bin/env python3

import sys
from joblib import dump, load
from sklearn.feature_extraction import DictVectorizer


def prepare_instances(xseq):
	features = []
	for interaction in xseq:
		token_dict = {feat.split('=')[0]:feat.split('=')[1] for feat in interaction[1:]}
		features.append(token_dict)
	return features


if __name__ == '__main__':

    # load leaned model and DictVectorizer
	model = load(sys.argv[1])
	v  = load(sys.argv[2]) 

	for line in sys.stdin:

		fields = line.strip('\n').split("\t")
		(sid,e1,e2) = fields[0:3]        
		vectors = v.transform(prepare_instances([fields[4:]]))
		prediction = model.predict(vectors)

		if prediction != "null" :            
			print(sid,e1,e2,prediction[0],sep="|")

