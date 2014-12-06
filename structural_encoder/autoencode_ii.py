import logging
import random
from filereader import WindowIterator
import theano.tensor as T
import theano
import numpy as np
import pinject
import filereader


class Autoencoder(object):
	def __init__(self,
				 word_vec_width,
				 batch_size,
				 num_hidden,
				 learning_rate=0.1):
		self.num_hidden = num_hidden
		self.learning_rate = learning_rate
		self.word_vec_width = word_vec_width
		self.batch_size = batch_size

		self.vocab_mat = T.fmatrix('vocab')
		self.word_onehot = T.fmatrix('word_onehot')
		b = T.fvector('b')
		W = T.fmatrix('W')
		f = 1 / (1 + T.exp(-(W * (self.word_onehot.dot(self.vocab_mat) + b))))
		s = T.sum(f)

		self.exec_fn = theano.function(
			[self.word_onehot, b, W, self.vocab_mat],
			f,
			allow_input_downcast=True)

		self.word_onehot_c = T.fmatrix('word_onehot_c')
		f_c = 1 / (1 + T.exp(-(W * (self.word_onehot_c.dot(self.vocab_mat)) + b)))
		s_c = T.sum(f_c)

		J = T.largest(0, 1 - s + s_c)
		self.grad = theano.grad(J, [b, W, self.vocab_mat])

		self.grad_fn = theano.function(
			[self.word_onehot, self.word_onehot_c, b, W, self.vocab_mat],
			self.grad,
			allow_input_downcast=True)

	def train_step(self,
				   document_iterator,
				   corruptor,
				   vocab_transformer,
				   b,
				   W,
				   vocab_matrix):
		assert isinstance(vocab_transformer, VocabularyVectorizer)
		dataset, corrupted_data = self.build_dataset(
			document_iterator,
			corruptor,
			vocab_transformer)

		[db, dW, dVocab] = self.grad_fn(dataset, corrupted_data, b, W, vocab_matrix)
		b_new = np.copy(b) - db * self.learning_rate
		W_new = np.copy(W) - dW * self.learning_rate
		vocab_new = np.copy(vocab_matrix) - dVocab * self.learning_rate

		return b_new, W_new, vocab_new


	def train(self, document_iterator):
		assert isinstance(document_iterator, IteratorAndVocab)
		vocab = VocabularyVectorizer(document_iterator)
		corruptor = Corruptor(document_iterator)
		W = np.random.random((500*self.word_vec_width, self.num_hidden))
		b = np.random.random((self.num_hidden,))
		vocab_matrix = np.random.random((vocab.vocab.size, self.word_vec_width))
		b, W, vocab_matrix = self.train_step(document_iterator, corruptor, vocab, b, W, vocab_matrix)



	def build_dataset(self,
					  document_source,
					  corruptor,
					  vocabulary_vectorizer):
		batch = []
		corrupted_batch = []
		assert isinstance(document_source, IteratorAndVocab)
		assert isinstance(corruptor, Corruptor)
		assert corruptor.document_iterator == document_source
		assert isinstance(vocabulary_vectorizer, VocabularyVectorizer)

		window_iter = iter(document_source)
		for i in xrange(self.batch_size):
			words = window_iter.next()
			corrupted_words = corruptor.corrupt(words)
			w_vec = np.concatenate([vocabulary_vectorizer[word]
									for word in words])
			cw_vec = np.concatenate([vocabulary_vectorizer[word]
									 for word in corrupted_words])
			batch.append(w_vec)
			corrupted_batch.append(cw_vec)
		batch_matrix = np.concatenate(batch, axis=1)
		corrupted_batch_matrix = np.concatenate(corrupted_batch, axis=1)
		return batch_matrix, corrupted_batch_matrix



class IteratorAndVocab(object):
	def __init__(self, fileiterator):
		self.fileiterator = fileiterator
		self.vocab = list({word for document in fileiterator for word in document})
		self.word_index = {word: ix for ix, word in enumerate(self.vocab)}
		self.size = len(self.vocab)

	def __iter__(self):
		return iter(self.fileiterator)

class VocabularyVectorizer(object):
	def __init__(self, vocab):
		assert isinstance(vocab, IteratorAndVocab)
		self.vocab = vocab

	def __getitem__(self, word):
		ix = self.vocab.word_index[word]
		vec = np.zeros((self.vocab.size,1))
		vec[ix] = 1
		return vec


class Corruptor(object):
	def __init__(self, document_source):
		assert isinstance(document_source, IteratorAndVocab)
		self.document_iterator = document_source

	def corrupt(self, document):
		"""
		:param document: the document to corrupt
		:return: List[String]: a copy of the document that we'll corrupt
		"""
		center = len(document) / 2
		new_doc = document[:]
		new_doc[center] = self.document_iterator.vocab[
			random.randint(0, len(self.document_iterator.vocab) - 1)]
		return new_doc


def main():
	obj_graph = pinject.new_object_graph()
	logging.warn("Creating iterator")
	iterator = obj_graph.provide(
		filereader.WindowIterator).configure('gutenberg/1/0/1/1016', 5)
	logging.warn("Creating vocab")
	vocab = IteratorAndVocab(iterator)
	logging.warn("Creating ae")
	ae = Autoencoder(500, 10000, 500)
	logging.warn("Training step")
	ae.train(vocab)


main()