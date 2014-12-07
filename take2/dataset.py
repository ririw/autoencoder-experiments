'''
	A dataset, which includes an interator that 
	walks over all documents.
'''
start_of_window = u'SOW'
end_of_window = u'EOW'

import scipy
import nltk
from nltk.tokenize import TreebankWordTokenizer
import fnmatch
import os

class DatasetIterator(object):
	def __init__(self,vocab):
		self.fileiter = vocab.fileiter
		self.vocab = vocab
	
	def batch_iter(self, batch_size=10000):
		window_iter = iter(self.fileiter)
		while True:
			yield self.batch_from_iter(window_iter, batch_size)

	def batch_from_iter(self, iterator, batch_size):
		batch = scipy.sparse.lil_matrix((batch_size, self.vocab.vocab_size * self.fileiter.window_size))
		for i in xrange(batch_size):
			window = iterator.next()
			for word in window:
				batch[i, self.vocab.vocab_index[word]] = 1.0
		return batch.tocsr()
			
class CorruptedPairBatch(object):
	def __init__(self, dataset_iterator):
		self.dataset_iterator = dataset_iterator

	def batch_iter(self, batch_size=10000):
		center = self.dataset_iterator.vocab.fileiter.window_size / 2
		centerstart = center * self.dataset_iterator.vocab.vocab_size
		centerend = (center + 1) * self.dataset_iterator.vocab.vocab_size
		for batch in self.dataset_iterator(batch_size):
			size = batch.shape[0]
			corrupt_batch = batch.copy()
			corrupt_batch[0, centerstart:centerend] = 0
			random_offsets = np.random.randint(0, self.dataset_iterator.vocab.vocab_size, size)
			for i in xrange(size):
				corrupt_batch[i, centerstart+random_offsets[i]] = 1.0
			yield batch, corrupt_batch
		
class Vocab(object):
	def __init__(self, fileiter):
		self.fileiter = fileiter
		self.vocab = {w for w in fileiter.word_iter()}
		self.vocab.add(start_of_window)
		self.vocab.add(end_of_window)
		self.vocab_index = {w: ix for ix, w in enumerate(self.vocab)}
		self.vocab_size = len(self.vocab)

class BasicFileIter(object):
	def __init__(self, file_or_dir, window_size=5):
		self.sentence_tokenizer = SentenceTokenizer()
		self.word_tokenizer = WordTokenizer()
		self.window_size = window_size
		self.file_or_dir = file_or_dir

	def word_iter(self):
		if isinstance(self.file_or_dir, str):
			file_iterator = self._get_text_file_matches(self.file_or_dir)
			sentence_iterator = self._sentence_iterator(file_iterator)
		else:
			file_iterator = iter(self.file_or_dir)
			sentence_iterator = self._sentence_iterator(self.file_or_dir)
		for scentence in sentence_iterator:
			for word in self.word_tokenizer.tokenize(scentence):
				yield word.lower()
	
	def __iter__(self):
		if isinstance(self.file_or_dir, str):
			file_iterator = self._get_text_file_matches(self.file_or_dir)
			sentence_iterator = self._sentence_iterator(file_iterator)
			return self._window_iterator(sentence_iterator)
		else:
			file_iterator = iter(self.file_or_dir)
			sentence_iterator = self._sentence_iterator(self.file_or_dir)
			return self._window_iterator(sentence_iterator)
		
	def _window_iterator(self, sentence_iterator):
		for sentence in sentence_iterator:
			window = [start_of_window for i in xrange(self.window_size)]
			for word in self.word_tokenizer.tokenize(sentence):
				lc_word = word.lower()
				window.pop(0)
				window.append(lc_word)
				# Yield a COPY
				yield window[:]
			for i in xrange(self.window_size-1):
				window.pop(0)
				window.append(end_of_window)
				# Yield a COPY
				yield window[:]

	def _sentence_iterator(self, file_match_iterator):
		for file in file_match_iterator:
			with file as dataset:
				for sentence in self.sentence_tokenizer.\
					tokenize(dataset.read().decode('utf8', 'ignore')):
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


#fi = BasicFileIter("./gutenberg/0")
#vc = Vocab(fi)
#ds = DatasetIterator(fi, vc)
#for batch in ds.batch_iter(100):
	#print batch.shape
	#print batch.sum()
