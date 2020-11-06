
import numpy
import tensorflow


# FROM https://github.com/branavg/Text-Generation/blob/master/Text_Generation.ipynb
def split_input_target (chunk):
    input_text = chunk [:-1]
    target_text = chunk [1:]
    return input_text, target_text

# FROM https://github.com/branavg/Text-Generation/blob/master/Text_Generation.ipynb
def prepare_batch_data (full_data, sequence_length, batch_size, buffer_size = 10000) :
  tf_dataset = tensorflow . data . Dataset . from_tensor_slices (full_data)
  tf_dataset = tf_dataset . batch (sequence_length + 1, drop_remainder = True)
  tf_dataset = tf_dataset . map (split_input_target)
  tf_dataset = tf_dataset . shuffle (buffer_size) . batch (batch_size, drop_remainder = True)
  return tf_dataset


def split_categories_input_target (chunk) :
    categories = chunk ['categories'] [ : -1 ]
    input_text = chunk ['tokens'] [ : -1 ]
    target_text = chunk ['tokens'] [ 1 : ]
    return ((categories, input_text), target_text)


def prepare_batch_data_with_categories (full_data, sequence_length, batch_size, buffer_size = 10000) :
  tf_dataset = tensorflow . data . Dataset . from_tensor_slices (full_data)
  tf_dataset = tf_dataset . batch (sequence_length + 1, drop_remainder = True)
  tf_dataset = tf_dataset . map (split_categories_input_target)
  tf_dataset = tf_dataset . shuffle (buffer_size) . batch (batch_size, drop_remainder = True)
  return tf_dataset




def repeat_categories (categories, sequence_len) :
  return numpy . tile (categories, sequence_len) . reshape ((sequence_len, categories . shape [0]))

def create_sequence_with_categories (token_sequence, categories) :
  sequence_len = len (token_sequence)
  categories_sequence = repeat_categoties (categories, sequence_len)
  return (token_sequence, categories_sequence)

































