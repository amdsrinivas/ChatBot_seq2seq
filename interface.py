import pickle
import string
import tensorflow as tf
import seq2seq_wrapper
from datasets.cornell_corpus import data
import data_utils
import numpy as np

def zero_pad(l):
	return np.array(l + [np.zeros(1).reshape(-1) for i in range(25-len(l))])
metadata = 'datasets/cornell_corpus/metadata.pkl'
puncs = list(string.punctuation)
print(puncs)

metadata, idx_q, idx_a = data.load_data(PATH='datasets/cornell_corpus/')
(trainX, trainY), (testX, testY), (validX, validY) = data_utils.split_dataset(idx_q, idx_a)
xseq_len = trainX.shape[-1]
yseq_len = trainY.shape[-1]
batch_size = 32
xvocab_size = len(metadata['idx2w'])
yvocab_size = xvocab_size
emb_dim = 1024

print('x_seq_len : {0}'.format(xseq_len))

idx2w = metadata['idx2w']
w2idx = metadata['w2idx']
print('0 : {0}'.format(idx2w[0]))

model = seq2seq_wrapper.Seq2Seq(xseq_len=xseq_len,
                               yseq_len=yseq_len,
                               xvocab_size=xvocab_size,
                               yvocab_size=yvocab_size,
                               ckpt_path='ckpt/cornell_corpus/',
                               emb_dim=emb_dim,
                               num_layers=3
                               )

print('Importing last session')
sess = model.restore_last_session()
print('Imported last session')

while(True):
	question = list(map(lambda x : ''.join(ch for ch in x if ch not in puncs), input("Say something :\n").strip().lower().split() ))
	#print(question)
	#question = filter_puncs(question)
	tokens = []
	for word in question:
		if word in w2idx.keys():
			tokens.append(np.array(w2idx[word], dtype=np.int16).reshape(-1))
	#print(tokens)
	tokens = zero_pad(tokens)
	#print(len(tokens))
	out = model.predict(sess,tokens)
	print('Output:\t')
	for word in out[0]:
		if word !=0 :
			print(idx2w[word],end="\t")
	print()
