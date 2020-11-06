import os
import re

import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

from model.model_lstm import *
from dirs import checkpoints_dir
from model.category_list import admissible_categories as category_list

checkpoint_dir = os.path.join('model', checkpoints_dir)

# model params

parameters_lstm_char = {
  'name' : 'lstm-char',
  'token_type' : 'char',
  'conditioned' : False,
  'sequence_length' : 112,
  'batch_size' : 64,
  'vocab_size' : 0,
  'embedding_dimension' : 256,
  'nb_lstm_layers' : 1,
  'lstm_layer_size' : 2048,
  'nb_dense_layers' : 1,
  'hidden_dense_layer_size' : 512,
}

parameters_lstm_word = {
  'name' : 'lstm-word',
  'token_type' : 'word',
  'conditioned' : False,
  'sequence_length' : 112,
  'batch_size' : 64,
  'vocab_size' : 0,
  'embedding_dimension' : 1024,
  'nb_lstm_layers' : 1,
  'lstm_layer_size' : 1024,
  'nb_dense_layers' : 1,
  'hidden_dense_layer_size' : 1024,
}

parameters_lstm_char_cond = {
  'name' : 'lstm-char-cond',
  'token_type' : 'char',
  'conditioned' : True,
  'category_min_size' : 4011,
  'sequence_length' : 112,
  'batch_size' : 8,
  'vocab_size' : 0,
  'embedding_dimension' : 256,
  'nb_lstm_layers' : 1,
  'lstm_layer_size' : 2048,
  'nb_dense_layers' : 1,
  'hidden_dense_layer_size' : 512,
  'categories_embedding_dimension' : 12,
}

parameters_lstm_word_cond = {
  'name' : 'lstm-word-cond',
  'token_type' : 'word',
  'conditioned' : True,
  'category_min_size' : 4011,
  'sequence_length' : 112,
  'batch_size' : 8,
  'vocab_size' : 0,
  'embedding_dimension' : 1024,
  'nb_lstm_layers' : 1,
  'lstm_layer_size' : 1024,
  'nb_dense_layers' : 1,
  'hidden_dense_layer_size' : 0,
  'categories_embedding_dimension' : 12,
}

# tokenizer

# very basic tokenizer
class LSTMTokenizer:

  def __init__(self, token_type):
    self.token_type = token_type
    
    if(token_type == 'char'):
      from model.lstm_char_tokenizer import char_to_index_dict as word_to_index_dict
      self.oov_placeholder = ' '
    else:
      from model.lstm_word_tokenizer import word_to_index_dict as word_to_index_dict
      self.oov_placeholder = '[?]'
    
    self.do_split = (token_type == 'word')
    
    self.word_to_index = word_to_index_dict
    self.index_to_word = { v : k for k, v in self.word_to_index.items() }
    self.index_to_word[0] = '<ZT>'
    
    self.join_string = '' if token_type == 'char' else ' '
    
  def encode_word(self, w):
    return self.word_to_index[w] if (w in self.word_to_index) else self.word_to_index[self.oov_placeholder]
  
  def tokenize(self, text):
    if(self.do_split):
      words = text.split()
    else:
      words = text
    return [ self.encode_word(w) for w in words ]
    
  def decode(self, sequence):
    #print(f'>>decode, sequence= {sequence}')
    words = [ self.index_to_word[t] for t in sequence ]
    return self.join_string.join(words)

# model

def load_model(parameters):
  if(parameters['conditioned']):
    model = ConditionedLSTMModel(vocab_size = parameters['vocab_size'],
                                 nb_categories = parameters['nb_categories'],
                                 sequence_length = parameters['sequence_length'],
                                 network_params = {
                                   'text_embedding_dimension' : parameters['embedding_dimension'],
                                   'categories_embedding_dimension' : parameters['categories_embedding_dimension'],
                                   'nb_lstm_layers' : parameters['nb_lstm_layers'],
                                   'lstm_size' : parameters['lstm_layer_size'],
                                   'nb_dense_layers' : parameters['nb_dense_layers'],
                                   'dense_layer_size' : parameters['hidden_dense_layer_size'],
                                   },
                                 batch_size = 1,
                                 lstm_dropout = 0.,
                                 checkpoint_directory = checkpoint_dir,
                                 model_name = parameters['model_name'])
  else:
    model = LSTMModel(vocab_size = parameters['vocab_size'],
                      sequence_length = parameters['sequence_length'],
                      nb_lstm_layers = parameters['nb_lstm_layers'],
                      embedding_dimension = parameters['embedding_dimension'],
                      lstm_size = parameters['lstm_layer_size'],
                      lstm_dropout = 0.,
                      last_lstm_layer_returns_sequence = True,
                      nb_dense_layers = parameters['nb_dense_layers'],
                      hidden_dense_layer_size = parameters['hidden_dense_layer_size'],
                      batch_size = 1,
                      model_name = parameters['model_name'],
                      checkpoint_directory = checkpoint_dir)
  model._is_in_generate_mode = True
  model.model.load_weights(tf.train.latest_checkpoint(model.checkpoint_directory)).expect_partial()
  return model


def prepare_tokenizer(parameters):
  if(parameters['token_type'] ==  'char'):
    eot_text = '*'
  else:
    eot_text = '<EOT>'
  tokenizer = LSTMTokenizer(parameters['token_type'])
  parameters['eot_text'] = eot_text
  parameters['vocab_size'] = len(tokenizer.word_to_index) + 1
  return tokenizer

def prepare_categories(parameters, categories):
  #admissible_categories = [ category for category, size in category_sizes if size >= parameters['...'] ]
  parameters['nb_categories'] = len(category_list)
  categories_input = np.zeros(parameters['nb_categories'], dtype='float32')
  try:
    hot_indexes = [ category_list.index(cat) for cat in categories ]
  except(ValueError):
    hot_indexes = []
  categories_input[hot_indexes] = 1.
  return categories_input

def generate_text(model, is_cond, tokenizer, prompt, length = 120, temperature = 1., categories = None, default_seed = [ 0, ]):
  seed_tokens = tokenizer.tokenize(prompt)
  if(len(seed_tokens) == 0):
    seed_tokens = default_seed
  if(is_cond):
    output_seq = model.generate(seed_tokens, categories, length, temperature = temperature)
  else:
    output_seq = model.generate(temperature, seed_tokens, length)
  output = tokenizer.decode(output_seq)
  return output

def lstm_generate(model_name, prompt, temperature = 1., categories = None):

  if(model_name == 'lstm-char'):
    parameters = parameters_lstm_char
  elif(model_name == 'lstm-word'):
    parameters = parameters_lstm_word
  elif(model_name == 'lstm-char-cond'):
    parameters = parameters_lstm_char_cond
  elif(model_name == 'lstm-word-cond'):
    parameters = parameters_lstm_word_cond
  else:
    print('>>>Error:lstm_generate:model_name')
    return '(Error)'
  parameters['model_name'] = model_name
    
  try:
    tokenizer = prepare_tokenizer(parameters)
  except(Exception):
    print('>>>Error:lstm_generate:prepare_tokenizer')
    return '(Error)'
  
  is_cond = (model_name[-4:] == 'cond')
  if(is_cond):
    categories_input = prepare_categories(parameters, categories)
  else:
    categories_input = None
  
  model = load_model(parameters)
  
  # check categories input
  output = generate_text(model, is_cond, tokenizer, prompt, temperature = temperature, categories = categories_input)
  
  output = re.sub(r' +', ' ', output)
  if(parameters['token_type'] == 'char'):
    output = re.sub('\* *', '\n', output)
  else:
    output = re.sub(r'(\[\?\])+ *', '\n', output)
  output = re.sub(r'\n+', '\n', output)
  
  return output

