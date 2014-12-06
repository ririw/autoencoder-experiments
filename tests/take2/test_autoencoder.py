import numpy as np
import theano 
import theano.tensor as T
from take2.dataset import BasicFileIter,Vocab,DatasetIterator,CorruptedPairBatch
from take2.autoencoder import AutoEncoder

def get_autoencoder(vocab_len=100):
	fi = BasicFileIter("./gutenberg/0")
	vc = Vocab(fi)
	ds = DatasetIterator(vc)
	corruptor = CorruptedPairBatch(ds)
	ae = AutoEncoder(vocab_len, 500, corruptor)
	return ae

def test_transform_function():
	ae = get_autoencoder()
	W = T.fmatrix()
	data = T.fmatrix()
	b = T.fvector()
	W_actual = np.random.random((113, 71))-0.5
	b_actual = np.random.random(71)-0.5
	data_actual = np.random.random((40, 113))-0.5
	
	#fn = theano.function([W, b, data], data.dot(W)+b, allow_input_downcast=True)
	fn = theano.function([W, b, data], ae.transform_function(W, b, data), allow_input_downcast=True)

	print(fn(W_actual, b_actual, data_actual))

def test_wordvec_transform_np():
	ae = get_autoencoder(4)
	ae.vocab_size = 3
	data_actual = np.array([
		[1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
		[1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
		[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
	])
	vocab_mat_actual = np.array([
		[2, 3],
		[4, 6],
		[0, 1]
	])
	print(ae.wordvec_np(data_actual, vocab_mat_actual))

def test_wordvec_transform():
	ae = get_autoencoder(4)
	ae.vocab_size = 3
	data = T.fmatrix()
	vocab = T.fmatrix()

	fn = theano.function([data, vocab], ae.wordvec_transform(data, vocab), allow_input_downcast=True)
	data_actual = np.array([
		[1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
		[1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
		[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
	])
	vocab_mat_actual = np.array([
		[2, 3],
		[4, 6],
		[0, 1]
	])
	print(fn(data_actual, vocab_mat_actual))

