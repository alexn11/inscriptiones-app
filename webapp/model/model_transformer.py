"""
Main source: rewriting and adaptation from the tutorial code:
  https://www.tensorflow.org/tutorials/text/transformer
"""


import os
os.environ ['TF_CPP_MIN_LOG_LEVEL'] = '3' 


import time



import numpy as np
import tensorflow as tf
import matplotlib.pyplot as pyplot

import tensorflow_datasets as tfds
tfds_SubwordTextEncoder = tfds.features.text.SubwordTextEncoder
from tensorflow.keras import layers as tfk_layers
from tensorflow.keras import optimizers as tfk_optimizers




# == position encoding ==

# the two following functions are exact copy/paste from main source

def get_angles(pos, i, d_model):
  angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
  return pos * angle_rates

def positional_encoding(position, d_model):
  angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                          np.arange(d_model)[np.newaxis, :],
                          d_model)
  # apply sin to even indices in the array; 2i
  angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
  # apply cos to odd indices in the array; 2i+1
  angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
  pos_encoding = angle_rads[np.newaxis, ...]
  return tf.cast(pos_encoding, dtype=tf.float32)

# == masking ==

def create_padding_mask(sequence):
  mask = tf.cast(tf.math.equal(sequence, 0), tf.float32)
  return mask[ :, tf.newaxis, tf.newaxis, : ]

def create_look_ahead_mask(size):
  mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
  return mask

def create_combined_padding_and_look_ahead_mask(sequence):
  padding_mask = create_padding_mask(sequence)
  look_ahead_mask = create_look_ahead_mask(tf.shape(sequence)[1])
  return tf.maximum(padding_mask, look_ahead_mask)


def create_masks(seq_input, seq_target):
  encoder_padding_mask = create_padding_mask(seq_input)
  decoder_padding_mask = create_padding_mask(seq_input)
  #look_ahead_mask = create_look_ahead_mask(tf.shape(seq_target)[1])
  #decoder_target_padding_mask = create_padding_mask(seq_target)
  decoder_target_mask = create_combined_padding_and_look_ahead_mask(seq_target)
  return encoder_padding_mask, decoder_padding_mask, decoder_target_mask



# == scaled dot attention ==

def scaled_dot_product_attention(q, k, v, mask):
  qk = tf.matmul(q, k, transpose_b=True)
  d_k = tf.cast(tf.shape(k)[ -1 ], tf.float32)
  scaled_attention_logits = qk / tf.math.sqrt(d_k)
  if (mask is not None):
    scaled_attention_logits += -1.e9 * mask
  attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
  output = tf.matmul(attention_weights, v)
  return output, attention_weights

# == multihead attention unit ==

class MultiheadAttention(tfk_layers.Layer):
  def __init__(self, nb_heads, head_dim):
    super(MultiheadAttention, self).__init__()
    self.nb_heads = nb_heads
    self.head_dim = head_dim
    self.d_model = self.head_dim * self.nb_heads
    self.Wq = tfk_layers.Dense(self.d_model)
    self.Wk = tfk_layers.Dense(self.d_model)
    self.Wv = tfk_layers.Dense(self.d_model)
    self.output_layer = tfk_layers.Dense(self.d_model)
  def split_heads(self, x, batch_size):
    x = tf.reshape(x, (batch_size, -1, self.nb_heads, self.head_dim))
    x = tf.transpose (x, perm=[ 0, 2, 1, 3 ])
    return x # shape = batch size, nb heads, sequence len, head dim
  def call(self, q, k, v, mask):
    batch_size = tf.shape(q)[0]
    q = self.split_heads(self.Wq(q), batch_size)
    k = self.split_heads(self.Wk(k), batch_size)
    v = self.split_heads(self.Wv(v), batch_size)
    z, attention_weights = scaled_dot_product_attention(q, k, v, mask)
    z = tf.reshape(tf.transpose(z, perm=[ 0, 2, 1, 3 ]), (batch_size, -1, self.d_model))
    output = self.output_layer(z)
    return output, attention_weights


# == feedforward layer ==

def create_pointwise_feedforward_unit(d_model, d_ff):
  return tf.keras.Sequential([ tfk_layers.Dense(d_ff, activation='relu'),
                               tfk_layers.Dense(d_model) ])
  
# == encoder layer ==

class EncoderLayer(tfk_layers.Layer):
  
  def __init__(self, nb_heads, head_dim, d_ff, dropout_rate=0.1):
    super(EncoderLayer, self).__init__()
    d_model = nb_heads * head_dim
    self.multihead_attention_layer = MultiheadAttention(nb_heads, head_dim)
    self.attention_dropout = tfk_layers.Dropout(dropout_rate)
    self.attention_normalization = tfk_layers.LayerNormalization(epsilon=1.e-6)
    self.feedforward_layer = create_pointwise_feedforward_unit(d_model, d_ff)
    self.output_dropout = tfk_layers.Dropout(dropout_rate)
    self.output_normalization = tfk_layers.LayerNormalization(epsilon=1.e-6)
    
  def call(self, x, mask, is_training):
    z, _ = self.multihead_attention_layer(x, x, x, mask)
    z = self.attention_dropout(z, training=is_training)
    attention_output = self.attention_normalization(x + z)
    z = self.feedforward_layer(attention_output)
    z = self.output_dropout(z, training=is_training)
    z = self.output_normalization(attention_output + z)
    return z 
    

# == decoder layer ==

class DecoderLayer(tfk_layers.Layer):
  
  def __init__(self, nb_heads, head_dim, d_ff, dropout_rate=0.1):
    super(DecoderLayer, self).__init__()
    d_model = nb_heads * head_dim
    self.decoder_input_attention_layer = MultiheadAttention(nb_heads, head_dim)
    self.decoder_input_dropout = tfk_layers.Dropout(dropout_rate)
    self.decoder_input_normalization = tfk_layers.LayerNormalization(epsilon=1.e-6)
    self.hidden_attention_layer = MultiheadAttention(nb_heads, head_dim)
    self.hidden_dropout = tfk_layers.Dropout(dropout_rate)
    self.hidden_normalization = tfk_layers.LayerNormalization(epsilon=1.e-6)
    self.feedforward_layer = create_pointwise_feedforward_unit(d_model, d_ff)
    self.output_dropout = tfk_layers.Dropout(dropout_rate)
    self.output_normalization = tfk_layers.LayerNormalization(epsilon=1.e-6)
    
  def call(self, x, encoder_output, padding_mask, look_ahead_mask, is_training):
    z, input_attention_weights = self.decoder_input_attention_layer(x, x, x, look_ahead_mask)
    z = self.decoder_input_dropout(z, training=is_training)
    q_decoder = self.decoder_input_normalization(x + z)
    z, hidden_attention_weights = self.hidden_attention_layer(q_decoder, encoder_output, encoder_output, padding_mask)
    z = self.hidden_dropout(z, training=is_training)
    attention = self.hidden_normalization(q_decoder + z)
    z = self.feedforward_layer(attention)
    z = self.output_dropout(z, training=is_training)
    output = self.output_normalization(attention + z)
    return output, input_attention_weights, hidden_attention_weights
    
# == encoder stack ==

class Encoder(tfk_layers.Layer):
  
  def __init__(self, nb_layers,vocab_size, max_pos, nb_heads, head_dim, d_ff, dropout_rate=0.1):
    super(Encoder, self).__init__()
    self.nb_layers = nb_layers
    self.d_model = nb_heads * head_dim
    self.embedding_layer = tfk_layers.Embedding(vocab_size, self.d_model)
    self.embedding_normalization_factor = tf.math.sqrt(tf.cast(self.d_model, tf.float32))
    self.pos_encoding = positional_encoding(max_pos, self.d_model)
    self.embedding_dropout = tfk_layers.Dropout(dropout_rate)
    self.encoder_layers = [ EncoderLayer(nb_heads, head_dim, d_ff, dropout_rate=dropout_rate) for l in range(self.nb_layers) ]
  
  def call(self, x, mask, is_training):
    sequence_len = tf.shape(x)[1]
    z = self.embedding_layer(x)
    z *= self.embedding_normalization_factor
    z += self.pos_encoding[ :, : sequence_len, : ]
    z = self.embedding_dropout(z, training=is_training)
    for encoder_layer in self.encoder_layers:
      z = encoder_layer(z, mask, is_training)
    return z

# == decoder stack ==

class Decoder(tfk_layers.Layer):
  
  def __init__(self, nb_layers, vocab_size, max_pos, nb_heads, head_dim, d_ff, dropout_rate=0.1):
    super(Decoder, self).__init__()
    self.nb_layers = nb_layers
    self.d_model = nb_heads * head_dim
    self.embedding_layer = tfk_layers.Embedding(vocab_size, self.d_model)
    self.embedding_layer_normalization_factor = tf.math.sqrt(tf.cast(self.d_model, tf.float32))
    self.pos_encoding = positional_encoding(max_pos, self.d_model)
    self.embedding_dropout = tfk_layers.Dropout(dropout_rate)
    self.decoder_layers = [ DecoderLayer(nb_heads, head_dim, d_ff, dropout_rate) for l in range(self.nb_layers) ]
    
  def call(self, x, encoder_output, padding_mask, look_ahead_mask, is_training):
    sequence_len = tf.shape(x)[1]
    attention_weights = {}
    z = self.embedding_layer(x) * self.embedding_layer_normalization_factor
    z += self.pos_encoding[ :, : sequence_len, : ]
    z = self.embedding_dropout(z, training = is_training)
    for decoder_i, decoder_layer in enumerate(self.decoder_layers):
      z, target_weights, hidden_weights = decoder_layer(z, encoder_output, padding_mask, look_ahead_mask, is_training)
      attention_weights[f'decoder_layer_{decoder_i}_target'] = target_weights
      attention_weights[f'decoder_layer_{decoder_i}_hidden'] = hidden_weights
    return z, attention_weights


class DecoderNoEmbedding(tfk_layers.Layer):
  
  def __init__(self, nb_layers, vocab_size, max_pos, nb_heads, head_dim, d_ff, dropout_rate=0.1):
    super(DecoderNoEmbedding, self).__init__()
    self.nb_layers = nb_layers
    self.d_model = nb_heads * head_dim
    self.pos_encoding = positional_encoding(max_pos, self.d_model)
    self.embedding_dropout = tfk_layers.Dropout(dropout_rate)
    self.decoder_layers = [ DecoderLayer(nb_heads, head_dim, d_ff, dropout_rate) for l in range(self.nb_layers) ]
    
  def call(self, x, mask, is_training):
    sequence_len = tf.shape(x)[1]
    attention_weights = {}
    z = x
    for decoder_i, decoder_layer in enumerate(self.decoder_layers):
      z, target_weights, hidden_weights = decoder_layer(z, x, mask, mask, is_training)
      attention_weights[f'decoder_layer_{decoder_i}_target'] = target_weights
      attention_weights[f'decoder_layer_{decoder_i}_hidden'] = hidden_weights
    return z, attention_weights


# == transformers ==

class Transformer(tf.keras.Model):
  
  def __init__(self, nb_layers, vocab_size, max_pos, nb_heads, head_dim, d_ff, dropout_rate=0.1):
    super(Transformer, self).__init__()
    self.encoder = Encoder(nb_layers, vocab_size, max_pos, nb_heads, head_dim, d_ff, dropout_rate=dropout_rate)
    self.decoder = Decoder(nb_layers, vocab_size, max_pos, nb_heads, head_dim, d_ff, dropout_rate=dropout_rate)
    self.output_layer = tfk_layers.Dense(vocab_size)
  
  def call(self, seq_input, input_padding_mask, seq_target, target_padding_mask, target_look_ahead_mask, is_training):
    encoder_output = self.encoder(seq_input, input_padding_mask, is_training)
    decoder_output, attention_weights = self.decoder(seq_target, encoder_output, target_padding_mask, target_look_ahead_mask, is_training)
    output = self.output_layer(decoder_output)
    return output, attention_weights
    

# Autoencoder version

class TransformerAutoregressive(tf.keras.Model):
  
  def __init__(self, nb_layers, vocab_size, max_pos, nb_heads, head_dim, d_ff, dropout_rate=0.1):
    super(TransformerAutoregressive, self).__init__()
    self.encoder = Encoder(nb_layers, vocab_size, max_pos, nb_heads, head_dim, d_ff, dropout_rate=dropout_rate)
    #self.decoder = DecoderNoEmbedding(nb_layers, vocab_size, max_pos, nb_heads, head_dim, d_ff, dropout_rate=dropout_rate)
    self.output_layer = tfk_layers.Dense(vocab_size)
  
  def call(self, seq_input, mask, is_training):
    encoder_output = self.encoder(seq_input, mask, is_training)
    #decoder_output, attention_weights = self.decoder(encoder_output, mask, is_training)
    #output = self.output_layer(decoder_output)
    output = self.output_layer(encoder_output)
    return output
    

















