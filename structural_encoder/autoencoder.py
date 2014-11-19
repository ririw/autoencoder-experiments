'''
	The actual autoencoder, yo.
'''

import theano.tensor as T
import theano
import logging
import pinject
import random
import numpy as np

class AutoEncoder(object):
	@pinject.copy_args_to_internal_fields
	def __init__(self, window_iterator, window_corruptor, vocab):
		self.configured = False
		self.vocab = vocab
		self.vocab_mat = T.fmatrix('vocab')
		# x has size num_samples x (window_size * vec_width)
		self.x = T.fmatrix('x')
		b = T.fvector('b')
		W = T.fmatrix('W')
		f = 1 / (1 + T.exp(-(W*(self.x.dot(self.vocab_mat) + b))))
		s = T.sum(f)

		self.exec_fn = theano.function([self.x, b, W, self.vocab_mat],
			f,
			allow_input_downcast=True)

		self.x_c = T.fmatrix('x_c')
		f_c = 1 / (1 + T.exp(-(W*(self.x_c.dot(self.vocab_mat)) + b)))
		s_c = T.sum(f_c)

		J = T.largest(0, 1 - s + s_c)
		self.grad = theano.grad(J, [b, W, self.vocab_mat])

		self.grad_fn = theano.function(
			[self.x, self.x_c, b, W, self.vocab_mat],
			self.grad,
			allow_input_downcast=True)

	def configure(self, batch_size, vocab_length, window_width, hidden_nodes):
		self._batch_size = batch_size
		self._vocab_length = vocab_length
		self._window_size = window_width
		self._hidden_nodes = hidden_nodes
		self.configured = True
		return self

	def _build_dataset(self, window_iterator, corrupted_iterator):
		assert self.configured
		batch = []
		corrupted_batch = []
		assert(window_iterator.window_size == self._window_size)
		window_iter = iter(window_iterator)
		corrupted_window_iter = iter(corrupted_iterator)
		for i in xrange(self._batch_size):
			words = window_iter.next()
			corrupted_words = corrupted_window_iter.next()
			w_vec  = np.concatenate([self._vocabulary[word] for word in words])
			cw_vec = np.concatenate([self._vocabulary[word] for word in corrupted_words])
			batch.append(w_vec)
			corrupted_batch.append(cw_vec)
		batch_matrix = np.concatenate(batch)
		corrupted_batch_matrix = np.concatenate(corrupted_batch)
		return batch_matrix, corrupted_batch_matrix
		# Test for result sizes,
		# Test for result datatypes
		# Test for result agreement except for corruption
	
	def train_step(self, window_iterator, corrupted_iterator, W, b, vocab):
		# Mock grad_fn with a much simpler function, that 
		# simply returns something silly (eg, assume all functions are x^2
		# and then see it goes to zero for all.
		import debug
		batch_matrix, corrupted_batch_matrix = \
			self._build_dataset(window_iterator, corrupted_iterator)
		
		[db, dW, dVocab] = self.grad_fn(batch_matrix, corrupted_batch_matrix, b, W, vocab)
		b_new = np.copy(b) - self.learning_rate * db
		W_new = np.copy(W) - self.learning_rate * dW
		vocab_new = np.copy(vocab) - self.learning_rate * dVocab

		return b_new, W_new, vocab_new

	def build_vocab_matrix(self):
		return np.random.random((len(self.vocab), self._vocab_length))

	def build_W_matrix(self):
		rows = self._hidden_nodes
		cols = self._vocab_length * self._hidden_nodes
		return np.random.random((rows, cols))

	def build_b_vector(self):
		return np.random.random(self._hidden_nodes)

	def transform(self, document):
		# Just mock a simple function to check this works
		assert(len(document) == self._window_size)
		w_vec  = np.concatenate([self._vocabulary[word] for word in document])
		[res] = self.exec_fn([w_vec])
		return res


class Vocab(object):
	def __init__(self, window_iterator):
		self._window_iterator = window_iterator
		self.vocab = list({words[0] for words in window_iterator})

	def check_against_iter(self, iterator):
		assert(self._window_iterator == iterator)

	def __len__(self):
		return len(self.vocab)

	def __getitem__(self, ix):
		self.vocab[ix]

class WindowCorruptor(object):
	def __init__(self, window_iterator, vocab):
		self._window_iterator = window_iterator
		self.vocab = vocab
		vocab.check_against_iter(window_iterator)

	def __iter__(self):
		return self._corrupted_iterator(iter(self._window_iterator))

	def iterate_forever(self):
		return self._corrupted_iterator(self._window_iterator.iterate_forever())

	def _corrupted_iterator(self, iterator):
		center = self._window_iterator.window_center()
		for window in iterator:
			window[center] = self.vocab[random.randint(0, len(self.vocab))]
			yield window


import filereader

obj_graph = pinject.new_object_graph()
iterator = obj_graph.provide(filereader.WindowIterator).configure('/home/riri/Downloads/gutenberg/0/0', 5)
vocab = Vocab(iterator)
corruptor = WindowCorruptor(iterator, vocab)
ae = AutoEncoder(iterator, corruptor, vocab)
ae.configure(50000, 500, 5, 500)
logging.warning('Commencing a training step')
b = ae.build_b_vector()
vocab_vec = ae.build_vocab_matrix()
W = ae.build_W_matrix()
ae.train_step(ae._window_iterator, ae._window_corruptor, W, b, vocab_vec)



