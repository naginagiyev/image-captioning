import re
from collections import Counter

class Vocabulary:
    def __init__(self, freqThresh=3):
        # min number of appearance of a word to be included to vocabulary
        self.freqThresh = freqThresh
        # stores words and their ids as value
        self.word2idx = {}
        # stores ids and the corresponding words as values
        self.idx2word = {}
        # count words so we'll not include less frequent words
        self.wordFreq = Counter()

        # special tokens
        self.PAD_TOKEN = '<pad>'
        self.START_TOKEN = '<start>'
        self.END_TOKEN = '<end>'
        self.UNK_TOKEN = '<unk>'
        
        self.word2idx[self.PAD_TOKEN] = 0
        self.word2idx[self.START_TOKEN] = 1
        self.word2idx[self.END_TOKEN] = 2
        self.word2idx[self.UNK_TOKEN] = 3
        
        self.idx2word[0] = self.PAD_TOKEN
        self.idx2word[1] = self.START_TOKEN
        self.idx2word[2] = self.END_TOKEN
        self.idx2word[3] = self.UNK_TOKEN
        
        self.idx = 4

    # function to split sentences into words and normalize them
    def tokenize(self, text):
        text = text.lower()
        text = re.sub(r'[^a-zA-Z0-9\']', ' ', text)
        words = text.split()
        return words
        
    # function that builds idx:word and word:idx pairs
    def buildVocabulary(self, captionsList):
        for caption in captionsList:
            words = self.tokenize(caption)
            self.wordFreq.update(words)
        
        for word, freq in self.wordFreq.items():
            if freq >= self.freqThresh:
                if word not in self.word2idx:
                    self.word2idx[word] = self.idx
                    self.idx2word[self.idx] = word
                    self.idx += 1

    # function to convert a text into tokens
    def numericalize(self, text):
        words = self.tokenize(text)
        return [self.word2idx.get(word, self.word2idx[self.UNK_TOKEN]) for word in words]

    def __len__(self):
        return len(self.word2idx)

    # returns the token of input word
    def __getitem__(self, word):
        return self.word2idx.get(word, self.word2idx[self.UNK_TOKEN])