"""
This version works only for documents with only alphabetical charaacters and spaces (no points etc)
0 = OOV
also: make sure there is consistence wrt b-string vs normal string
"""



def load_from_file(file_path):
  word_list_file = open(file_path, 'rb')
  word_list = []
  for line in word_list_file:
    word = line.strip()
    if(word == ''):
      continue
    word_list.append(word)
  return SimpleTokenizer(vocab = word_list, do_ignore_0_token = True)


def build_from_corpus(corpus):
  raise Exception('TODO')


class SimpleTokenizer():

  def __init__(self, vocab = None, do_ignore_0_token = False):
    if(vocab is None):
      self.vocab_size = 0
      return
    self.vocab = vocab
    self.word_to_index = { w : i+1 for i, w in enumerate(vocab) }
    self.vocab_size = len(self.vocab) + 1
    self.oov_token = 0
    self.oov_word = b'<UNK>'
    self.do_ignore_0_token = do_ignore_0_token
    
  def convert_word_to_token(self, word):
    try:
      token = self.word_to_index[word]
    except(KeyError):
      print(f'SimpleTokenizer.convert_word_to_token: unknown word: {word}')
      token = self.oov_token
    return token
    
  def encode(self, document):
    document_words = document.split()
    sequence = [ self.convert_word_to_token(word) for word in document_words ]
    return sequence
    
  def convert_token_to_word(self, token):
    if(token == 0):
      print(f'SimpleTokenizer.convert_token_to_word: token 0')
      if(self.do_ignore_0_token):
        word = b''
      else:
        word = self.oov_word
    elif(token > self.vocab_size):
      print(f'SimpleTokenizer.convert_token_to_word: unkown token {token}')
      word = self.oov_word
    else:
      word = self.vocab[token - 1]
    return word
    
  def decode(self, sequence):
    seq_len = len(sequence)
    if(seq_len == 0):
      return b''
    word_list = seq_len * [ None, ]
    for i, token in enumerate(sequence):
      word_list[i] = self.convert_token_to_word(token)
    return b' '.join(word_list)
    
  def save_to_file(self, file_path):
    raise Exception('TODO')






























