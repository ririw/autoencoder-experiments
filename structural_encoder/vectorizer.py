'''
	Build up the vocab, by first running over the iterator to
	work out what the vocab is, and then picking out the relevant
	vector from a dictionary.
'''
import numpy as np

class Vocabulary(object):
	def __init__(self, window_iterator):
		self.vocab = dict()
		word_counter = 0
		for w in window_iterator:
			if w not in self.vocab:
				self.vocab[w] = word_counter
				word_counter += 1
	
	def __getitem__(self, token):
		result = np.zeros((self.word_counter,))
		result[self.vocab[token]] = 1.0
		return result

	def __setitem__(self, token, value):
		self.vocab[token] = value
	
	def __delitem__(self, key):
		return self.vocab.__delitem__(key)

		
