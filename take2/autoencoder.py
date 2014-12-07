import theano.tensor as T
import scipy
import theano
import numpy as np

class AutoEncoder(object):
	def __init__(self, num_hidden, trasformed_vocab_size, dataset):
		self.trasformed_vocab_size = trasformed_vocab_size
		self.num_hidden = num_hidden
		self.dataset = dataset
		self.vocab = dataset.dataset_iterator.vocab

		self.window_size = dataset.dataset_iterator.vocab.fileiter.window_size
		self.vocab_size = dataset.dataset_iterator.vocab.vocab_size

		self.vocab_mat_actual = np.random.random((dataset.dataset_iterator.vocab.vocab_size, trasformed_vocab_size))-0.5
		self.W_actual = np.random.random((self.window_size*trasformed_vocab_size, num_hidden))-0.5
		self.b_actual = np.random.random(num_hidden)-0.5
		self.setup_theano()

	def setup_theano(self):
		self.vocab_mat = T.fmatrix('vocab')
		self.sample = T.fmatrix('sample')
		b = T.fvector('b')
		W = T.fmatrix('W')
		f = self.transform_function(
			W, 
			b, 
			self.wordvec_transform(self.sample, self.vocab_mat))
		s = T.sum(f)

		self.corrupt_sample = T.fmatrix('corrupt-sample')
		f_corrupt = self.transform_function(
			W,
			b,
			self.wordvec_transform(self.corrupt_sample, self.vocab_mat))
		s_corrupt = T.sum(f_corrupt)
		J = T.largest(0, 1 - s + s_corrupt)
		self.grad = theano.grad(J, [b, W, self.vocab_mat])

		self.grad_fn = theano.function(
			[self.sample, self.corrupt_sample, b, W, self.vocab_mat],
			self.grad,
			allow_input_downcast=True)

		self.exec_fn = theano.function([self.sample, b, W, self.vocab_mat],
			f,
			allow_input_downcast=True)

	def wordvec_transform(self, data_matrix, vocab_mat):
		results = []
		for i in range(self.window_size):
			sub_window = data_matrix[:, self.vocab_size*i:self.vocab_size*(i+1)]
			#sub_wordspace_window = vocab_mat.dot(sub_window)
			sub_wordspace_window = sub_window.dot(vocab_mat)
			results.append(sub_wordspace_window)
		return T.concatenate(results, axis=1)
		
	def wordvec_np(self, data_matrix, vocab_mat):
		results = []
		for i in range(self.window_size):
			sub_window = data_matrix[:, self.vocab_size*i:self.vocab_size*(i+1)]
			print np.sum(sub_window)
			sub_wordspace_window = sub_window.dot(vocab_mat)
			results.append(sub_wordspace_window)
		return np.concatenate(results, axis=1)
	
	def transform_function(self, W, b, data):
		return 1 / (1 + T.exp(-(data.dot(W) + b)))

	def transform_np(self, W, b, data):
		return 1 / (1 + np.exp(-(data.dot(W) + b)))

	def train(self, corrupted_paired_iterator):
		pass

	def execute(self, window):
		assert(len(window) == self.window_size)
		#data = scipy.sparse.lil_matrix((1, self.vocab.vocab_size * self.window_size))
		data = np.zeros((1, self.vocab.vocab_size * self.window_size))
		for window_offset, word in enumerate(window):
			ix = self.vocab.vocab_index[word]
			data[0, window_offset*self.trasformed_vocab_size + ix] = 1.0
		return self.exec_fn(data, self.b_actual, self.W_actual, self.vocab_mat_actual)
			














