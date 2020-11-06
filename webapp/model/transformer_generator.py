import os
import re

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

import matplotlib.pyplot as pyplot

from model.model_transformer import *

import model.simple_tokenizer as simple_tokenizer

from dirs import tokenizers_dir

tokenizers_dir = os.path.join('model', tokenizers_dir)

import importlib
parameter = None



def load_tokenizer(is_char_level, tokenizer_type, tokenizer_file_name):
  tokenizer_file_path = os.path.join(tokenizers_dir, tokenizer_file_name)
  if(not is_char_level):
    if(tokenizer_type == 'tfds'):
      tokenizer = tfds_SubwordTextEncoder.load_from_file(tokenizer_file_path)
    elif(tokenizer_type == 'house'):
      tokenizer = simple_tokenizer.load_from_file(tokenizer_file_path)
  else:
    tokenizer = tfds.features.text.ByteTextEncoder()
  vocab_size = tokenizer.vocab_size
  start_token = vocab_size
  end_token = vocab_size
  return tokenizer, vocab_size, start_token, end_token

def encode_sentence_compat_v6(sentence):
  return [ [ start_token, ] + tokenizer.encode(sentence.numpy()) ]

def encode_sentence(sentence, is_complete = True):
  encoded_sentence =  tokenizer.encode(sentence.numpy())
  if(is_complete):
    encoded_sentence.append(end_token)
  return [ encoded_sentence ]


def encode_sentence_for_tf_dataset(sentence):
  if(do_compat_v6):
    [ encoded_sentence ] = tf.py_function(encode_sentence_compat_v6, [ sentence, ], Tout=[ tf.int64, ])
  else:
    [ encoded_sentence ] = tf.py_function(encode_sentence, [ sentence, ], Tout=[ tf.int64, ])
  encoded_sentence.set_shape([ None, ])
  return encoded_sentence


def reshape_predictions(predictions):
  predictions = predictions[ 0, -1, : ]
  return predictions





def _set_compat_v6(compat_v6):
  global do_compat_v6
  do_compat_v6 = compat_v6






# == generate ==



def generate(transformer,
             tokenizer,
             start_token,
             end_token,
             seq_max_len,
             input_sentence,
             do_stop_at_end_token = True,
             temperature = 1.,
             default_seed = [ 1, ]):
  seq_input = tokenizer.encode(input_sentence)
  input_len = len(seq_input)
  if(input_len == 0):
    seq_input = default_seed
    input_len = len(seq_input)
  if(do_compat_v6):
    encoder_input = tf.expand_dims([ start_token, ] + seq_input, 0)
  else:
    encoder_input = tf.expand_dims(seq_input, 0)
  output = seq_input
  #attention_weights = []
  for i in range(input_len, seq_max_len):
    mask = create_combined_padding_and_look_ahead_mask(encoder_input)
    predictions = transformer(encoder_input, mask, is_training = False)
    predictions = reshape_predictions(predictions)
    if(temperature > 0.):
      logits = predictions / temperature
      predicted_token_index = tf.random.categorical(tf.expand_dims(logits, 0), num_samples=1)[ 0, 0 ].numpy()
    else:
      #GREEDY PREDICTION:
      predicted_token_index = tf.argmax(predictions, axis=-1).numpy()
    #predicted_token = valid_tokens[predicted_token_index]
    predicted_token = predicted_token_index
    if ((i > 0) and do_stop_at_end_token and (predicted_token in [ end_token, start_token ])):
      output.append(predicted_token)
      break
    output.append(predicted_token)
    encoder_input = tf.concat([ encoder_input, [[ predicted_token, ]] ], axis=-1)
  return output



def list_tokens(logit_len = None, token_list = None):
  if(token_list is None):
    labels = logit_len * [ None, ]
    for token in range(logit_len):
      try:
        decoded_token = tokenizer.decode([token])
      except(ValueError):
        decoded_token = '(OOV)'
      except(UnicodeDecodeError):
        decoded_token = '(INVALID)'
      else:
        if(decoded_token not in valid_chars):
          decoded_token = '(OOV+)'
      labels[token] = decoded_token
  else:
    labels = list(tokenizer.decode(token_list))
  return labels




def decode_generated_sequence(tokenizer, vocab_size, start_token, end_token, sequence, do_use_b_string = True):
  text = b'' if(do_use_b_string) else ''
  text_at_end = b'<ET>\n' if(do_use_b_string) else '<ET>\n'
  text_at_start = b'<ST>' if(do_use_b_string) else '<ST>'
  tokens = []
  for t in sequence:
    if(t == end_token):
      text += tokenizer.decode(tokens) + text_at_end
      tokens = []
    elif(t == start_token):
      text += text_at_start
    elif(t < vocab_size): # any other token is ignored
      tokens.append(t)
  if(tokens != []):
    text += tokenizer.decode(tokens)
  return text



def generate_text(model_version, prompt, temperature = 1.):
  global parameters
  parameters = importlib.import_module(f'model.transformer_parameters.parameters_v{model_version}')
  tokenizer, vocab_size, start_token, end_token = load_tokenizer(parameters.is_char_level,
                                                                 parameters.tokenizer_type,
                                                                 parameters.tokenizer_file_name)
  _set_compat_v6(parameters.do_compat_v6)
  transformer_model = TransformerAutoregressive(parameters.nb_layers,
                                                vocab_size + 2,
                                                parameters.seq_max_len,
                                                parameters.nb_heads,
                                                parameters.head_dim,
                                                parameters.d_ff)
  checkpoint = tf.train.Checkpoint(model=transformer_model)
  checkpoint_manager = tf.train.CheckpointManager(checkpoint, parameters.checkpoint_path, max_to_keep=5)
  checkpoint.restore(checkpoint_manager.latest_checkpoint).expect_partial()
  output = generate(transformer_model, tokenizer, start_token, end_token, parameters.seq_max_len, prompt.encode(), temperature = temperature)
  generated_text = decode_generated_sequence(tokenizer, vocab_size, start_token, end_token, output)
  generated_text = generated_text.decode('utf-8', errors = 'replace')
  generated_text = re.sub('^ +', '', generated_text)
  generated_text = re.sub(' +', ' ', generated_text)
  return generated_text




















