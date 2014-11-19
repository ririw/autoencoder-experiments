'''
	Read out our data files into a bunch of batches. Reading is lazy, and yields
	an interable of the file's contents, in the form of windows over the file. 
	start or end overflow is represented by the "SOW" and "EOW" strings.

	We segment with nltk.
'''

import fnmatch
import os
import pinject
import nltk.tokenize.punkt
from nltk.tokenize import TreebankWordTokenizer

start_of_window = u'SOW'
end_of_window = u'EOW'

class WindowIterator(object):
	@pinject.copy_args_to_internal_fields
	def __init__(self, sentence_tokenizer, word_tokenizer):
		self.configured = False

	def configure(self, file_or_dir, window_size):
		self._file_or_dir = file_or_dir
		self.window_size = window_size
		self.configured = True
		return self

	def window_center(self):
		assert(self.configured)
		return self.window_size / 2

	def __iter__(self):
		'''Iterate over every ".txt" file in directory'''
		assert(self.configured)
		if isinstance(self._file_or_dir, str):
			file_iterator = self._get_text_file_matches(self._file_or_dir)
			sentence_iterator = self._sentence_iterator(file_iterator)
			return self._window_iterator(sentence_iterator, self.window_size)
		else:
			file_iterator = iter(self._file_or_dir)
			sentence_iterator = self._sentence_iterator(self._file_or_dir)
			return self._window_iterator(sentence_iterator, self.window_size)

	def iterate_forever(self):
		'''
		iterate forever. You may send "true" to this iterator
		to stop the iteration
		'''
		exit = False
		iterator = iter(self)
		while not exit:
			# This should get the next value along, and then yield
			# to the consumer. If we exhaust the single-pass iterator
			# then we may simply reset the iterator, and on the next
			# loop-pass around the yield will happen successfully. The
			# consumer will never notice a differnce.
			try:
				next = iterator.next()
				exit = yield next
			except StopIteration:
				iterator = iter(self)
				
			
	
	def _window_iterator(self, sentence_iterator, window_size):
		for sentence in sentence_iterator:
			window = [start_of_window for i in xrange(window_size)]
			for word in self._word_tokenizer.tokenize(sentence):
				window.pop(0)
				window.append(word)
				# Yield a COPY
				yield window[:]
			for i in xrange(window_size-1):
				window.pop(0)
				window.append(end_of_window)
				# Yield a COPY
				yield window[:]

	def _sentence_iterator(self, file_match_iterator):
		for file in file_match_iterator:
			with file as dataset:
				for sentence in self._sentence_tokenizer.tokenize(dataset.read().decode('utf8', 'ignore')):
					yield sentence

	def _get_text_file_matches(self, directory):
		'''Returns an iterator of file-like objects'''
		for root, dirnames, filenames in os.walk(directory):
			for filename in fnmatch.filter(filenames, '*.txt'):
				if 'index' in filename.lower():
					continue
				yield open(os.path.join(root, filename))


class SentenceTokenizer(object):
	def __init__(self):
		self.sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
	def tokenize(self, document):
		return self.sent_detector.tokenize(document)

class WordTokenizer(object):
	def __init__(self):
		self._word_tokenizer = TreebankWordTokenizer()
	def tokenize(self, document):
		return self._word_tokenizer.tokenize(document)

