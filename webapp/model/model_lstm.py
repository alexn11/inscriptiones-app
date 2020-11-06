
import numpy
import os
import re

os.environ ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow
from tensorflow.keras import Model as tfk_Model
import tensorflow.keras.activations as tfk_activations
import tensorflow.keras.layers as tfk_layers
import tensorflow.keras.losses as tfk_losses
import tensorflow.keras.metrics as tfk_metrics
import tensorflow.keras.optimizers as tfk_optimizers

from model.train_data_pipeline import repeat_categories




class LSTMKerasModel (tfk_Model) :
  def __init__ (self,
                input_dim = 3,
                output_dim = 3,
                sequence_length = 18,
                batch_size = 1,
                embedding_dimension = 256,
                nb_lstm_layers = 4,
                lstm_size = 1024,
                lstm_dropout = 0.1,
                last_lstm_layer_returns_sequence = True,
                nb_dense_layers = 2,
                hidden_dense_layer_size = 1024) :
                
    #print (f'params:inputdim={input_dim},outpudim={output_dim},seqlen={sequence_length},batchsz={batch_size},embdim={embedding_dimension},')
    #print(f'nblstm={nb_lstm_layers},lstmsz={lstm_size},lstmdo={lstm_dropout},retseq={last_lstm_layer_returns_sequence},nbden={nb_dense_layers},densz={hidden_dense_layer_size}')
    
    super (LSTMKerasModel, self).__init__ ()
    
    self.nb_layers = nb_lstm_layers + nb_dense_layers
    self.nb_layers += 1
      
    self.layer_list = [ ]
    
    
    layer_index = 0
    
    single_feature_dim = 1
    current_feature_shape = (batch_size, sequence_length)
    
    self.layer_list.append (tfk_layers.Embedding (input_dim, embedding_dimension, name = 'text_embedding'))
    single_feature_dim = embedding_dimension
    current_feature_shape = (batch_size, sequence_length, single_feature_dim)
    layer_index += 1
      
    for lstm_layer_index in range (nb_lstm_layers) :
      is_first_lstm_layer = (lstm_layer_index == 0)
      is_last_lstm_layer = (lstm_layer_index == nb_lstm_layers - 1)
      self.layer_list.append (tfk_layers.LSTM (lstm_size,
                                                     dropout = lstm_dropout,
                                                     batch_input_shape = current_feature_shape,
                                                     #recurrent_initializer = 'glorot_uniform',
                                                     stateful = True,
                                                     return_sequences = last_lstm_layer_returns_sequence or not is_last_lstm_layer,
                                                     name = f'lstm_{lstm_layer_index}'))
      current_feature_shape = (batch_size, lstm_size, single_feature_dim)
    layer_index += nb_lstm_layers
    
    for i in range (nb_dense_layers - 1) :
      self.layer_list.append (tfk_layers.Dense (hidden_dense_layer_size, activation = 'relu', name = f'dense_{i}'))
    self.layer_list.append (tfk_layers.Dense (output_dim, name = 'dense_output'))
                                                             

                                                             
  def call (self, inputs) :
    x = inputs
    for layer in self.layer_list :
      x = layer (x)
    y = x
    #y = tfk_activations.softmax (x)
    return y





class ConditionedLSTMKerasModel (tfk_Model) :

  def __init__ (self,
                network_params,
                batch_size = 1,
                lstm_dropout = 0.1) :
                
    super (ConditionedLSTMKerasModel, self).__init__ ()
    
    self.parameters = network_params
    
    self.text_embedding = tfk_layers.Embedding (network_params ['vocab_size'],
                                                    network_params ['text_embedding_dimension'],
                                                    name = 'text_embedding')

    self.categories_embedding = tfk_layers.Dense (network_params ['categories_embedding_dimension'],
                                                      activation = None,
                                                      input_shape = (batch_size, network_params ['nb_categories']))
    # TODO: make this parametrizable:
    self.categories_activation = tfk_layers.ReLU (name = 'categories_activation')
    
    self.concatenator = tfk_layers.Concatenate (axis = -1, name = 'feat_concatenate')
    concatenated_dimension = network_params ['categories_embedding_dimension'] + network_params ['text_embedding_dimension']
    
    self.common_layers = []
    
    batch_input_shape = (batch_size, None, concatenated_dimension) 
    for i in range (network_params ['nb_lstm_layers']) :
      new_layer = tfk_layers.LSTM(network_params ['lstm_size'],
                                  dropout = lstm_dropout,
                                  batch_input_shape = batch_input_shape,
                                  stateful = True,
                                  return_sequences = True,
                                  name = f'lstm_{i}')
      self.common_layers.append(new_layer)
      batch_input_shape = (batch_size, network_params ['lstm_size'], concatenated_dimension) 
    for i in range(network_params['nb_dense_layers'] - 1) :
      new_layer = tfk_layers.Dense(network_params ['dense_layers_size'],
                                   activation = 'relu',
                                   name = f'dense_{i}')
      self.common_layers.append (new_layer)
      
    new_layer = tfk_layers.Dense (network_params ['vocab_size'],
                                    activation = 'relu',
                                    name = f'dense_output')
    self.common_layers.append (new_layer)


    #self.layers_output = []
    #for i in range (network_params ['nb_output_dense_layers']) :
    # TODO: add this later
    return

  def call (self, inputs) :
    categories_input = inputs [0]
    text_input = inputs [1]
    x0 = self.categories_embedding (categories_input)
    
    x1 = self.text_embedding (text_input)
    x = self.concatenator ([x0, x1])
    for layer in self.common_layers :
      x = layer (x)
    #y = tfk_activations.softmax (x)
    y = x
    return y








class LSTMModel :

  def __init__ (self,
                vocab_size = 54,
                sequence_length = None,
                embedding_dimension = 256,
                nb_lstm_layers = 1,
                lstm_size = 1024,
                lstm_dropout = 0.1,
                last_lstm_layer_returns_sequence = True,
                nb_dense_layers = 1,
                hidden_dense_layer_size = 0,
                loss = 'cross_entropy',
                batch_size = None,
                model_name = None,
                checkpoint_directory = './checkpoints') :
                
    self._is_compiled = False
    self._is_in_generate_mode = False
    self.batch_size = batch_size
    self.sequence_length = sequence_length
    
    if (model_name is None):
      model_name = 'unnamed'
    self.model_name = model_name
    self.checkpoint_directory = os.path.join(checkpoint_directory, self.model_name)
    
    self.input_batch_shape = (batch_size, sequence_length)
      
    self.model_params = {
      'input_dim' : vocab_size,
      'output_dim' : vocab_size,
      'sequence_length' : sequence_length,
      'nb_lstm_layers' : nb_lstm_layers,
      'lstm_dropout' : lstm_dropout,
      'nb_dense_layers' : nb_dense_layers,
      'hidden_dense_layer_size' : hidden_dense_layer_size,
      'last_lstm_layer_returns_sequence' : last_lstm_layer_returns_sequence,
      'embedding_dimension' : embedding_dimension,
    }
                                   
    self.model = LSTMKerasModel(batch_size = batch_size, ** self.model_params)

    #self.model.build (input_shape = self.input_batch_shape)
    self.optimizer = tfk_optimizers.Adam()
    self.change_loss(loss)
      
  #def eval_loss (self, labels, logits) :
  #  return tensorflow.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)
      
  def change_loss (self, loss) :
    self.loss_function_name = loss
    if (loss == 'mae') :
      self.loss_function = tfk_losses.MeanAbsoluteError ()
    elif (loss == 'cross_entropy') :
      self.loss_function = tfk_losses.SparseCategoricalCrossentropy (from_logits = True)
    elif (loss == 'mse') :
      self.loss_function = tfk_losses.MeanSquaredError ()
    else :
      raise Exception ('loss should be either \'cross_entropy\', \'mae\' or \'mse\'')
    #self.loss_function = self.eval_loss
    
      
  def __call__ (self, inputs) :
    return self.model (inputs)
        
  def _rebuild_train_model (self) :
    self.model = LSTMKerasModel (** self.model_params, batch_size = None)
    self.model.load_weights (tensorflow.train.latest_checkpoint (self.checkpoint_directory))
    self._is_in_generate_mode = False
        
  def train (self, nb_epochs = 1, nb_steps_per_epoch = 1, train_dataset = None, checkpoint_name_prefix = 'unnamed') :
    if (self._is_in_generate_mode) :
      self._rebuild_train_model ()
    self.checkpoint_name_prefix = checkpoint_name_prefix
    self.model.compile (loss = self.loss_function, optimizer = self.optimizer)
    checkpoint_callback = tensorflow.keras.callbacks.ModelCheckpoint (filepath = os.path.join (self.checkpoint_directory, self.checkpoint_name_prefix),
                                                                            save_weights_only = True)
    history = self.model.fit (train_dataset.repeat (),
                                  epochs = nb_epochs,
                                  steps_per_epoch = nb_steps_per_epoch,
                                  callbacks = [ checkpoint_callback, ])
    return history
  
  def set_generate_mode (self) :
    if (self._is_in_generate_mode) :
      return
    self.model = LSTMKerasModel (batch_size = 1, ** self.model_params)
    self.model.load_weights (tensorflow.train.latest_checkpoint (self.checkpoint_directory))
    self.model.build (tensorflow.TensorShape ( [1, None ]))
    self._is_in_generate_mode = True
    
  def set_training_mode (self) :
    raise Exception ('TODO')

  def eval (self, eval_metrics) :
    raise Exception ('TODO')
    losses = { metric_name : 0. for metric_name in eval_metrics }
    batch_count = 0
    do_xent = 'cross_entropy' in eval_metrics
    do_mae = 'mae' in eval_metrics
    do_mse = 'mse' in eval_metrics
    for feature_batch, label_batch in self.test_dataset :
      batch_count += 1
      predictions = self.model (feature_batch)
      if (do_xent) :
        losses ['cross_entropy'] += numpy.average (tfk_losses.categorical_crossentropy (label_batch, predictions, from_logits = True).numpy ())
      if (do_mae) :
        losses ['mae'] += numpy.average (tfk_losses.mean_absolute_error (label_batch, predictions).numpy ())
      if (do_mse) :
        losses ['mse'] += numpy.average (tfk_losses.mean_squared_error (label_batch, predictions).numpy ())
    return { k : v / batch_count for k, v in losses.items () }
    
  def generate (self, temperature, seed_tokens, length) :
  
    seed_len = len (seed_tokens)
    total_len = seed_len + length
    generated_tokens = numpy.zeros (total_len, dtype = 'int32')
    generated_tokens [ : seed_len ] = seed_tokens
  
    inputs = tensorflow.expand_dims (seed_tokens, 0)
    
    for i in range (length) :
      logits = self.model (inputs) [ :, -1, : ]
      logits = logits / temperature
      generated_token = tensorflow.squeeze (tensorflow.random.categorical (logits, num_samples = 1)).numpy ()
      generated_tokens [i + seed_len] = generated_token
      inputs = tensorflow.expand_dims ([ generated_token ], 0)
    
    return generated_tokens


  def save_weights (self, path) :
    self.model.save_weights (path)

  def load_weights (self, path) :
    self.model.load_weights (path)





class ConditionedLSTMModel :

  def __init__ (self,
                vocab_size = 54,
                nb_categories = 21,
                sequence_length = None,
                network_params = {
                  'text_embedding_dimension' : 256,
                  'categories_embedding_dimension' : 5,
                  'nb_lstm_layers' : 1,
                  'lstm_size' : 1024,
                  'nb_dense_layers' : 1,
                  'dense_layers_size' : 1024,
                },
                batch_size = None,
                lstm_dropout = 0.1,
                loss = 'cross_entropy',
                checkpoint_directory = './checkpoints',
                model_name = None) :
                
    self._is_compiled = False
    self._is_in_generate_mode = False
    
    self.vocab_size = vocab_size
    self.nb_categories = nb_categories
    self.sequence_length = sequence_length
    self.batch_size = batch_size
    self.lstm_dropout = lstm_dropout
    
    if (model_name is None) :
      model_name = 'unnamed'
    self.model_name = model_name
    self.checkpoint_directory = os.path.join(checkpoint_directory, self.model_name)
    
    self.network_params = { k : v for k, v in network_params.items () }
    self.network_params ['vocab_size'] = self.vocab_size
    self.network_params ['output_dim'] = self.vocab_size
    self.network_params ['nb_categories'] = self.nb_categories
        
    self.text_input_batch_shape = (batch_size, sequence_length)
    self.categories_input_batch_shape = (batch_size, nb_categories)
                                    
    self.model = ConditionedLSTMKerasModel (self.network_params, batch_size = self.batch_size, lstm_dropout = self.lstm_dropout)

    #self.model.build (input_shape = self.input_batch_shape)
    self.optimizer = tfk_optimizers.Adam ()
    self.change_loss (loss)
      
  #def eval_loss (self, labels, logits) :
  #  return tensorflow.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)
      
  def change_loss (self, loss) :
    self.loss_function_name = loss
    if (loss == 'mae') :
      self.loss_function = tfk_losses.MeanAbsoluteError ()
    elif (loss == 'cross_entropy') :
      self.loss_function = tfk_losses.SparseCategoricalCrossentropy (from_logits = True)
    elif (loss == 'mse') :
      self.loss_function = tfk_losses.MeanSquaredError ()
    else :
      raise Exception ('loss should be either \'cross_entropy\', \'mae\' or \'mse\'')
    #self.loss_function = self.eval_loss
    
 
  # THIS COULD BE FACTORED IN AN ABSTRACT CLASS    
  def __call__ (self, categories_inputs, text_inputs) :
    return self.model ((categories_inputs, text_inputs))
        
  # THIS COULD BE FACTORED IN AN ABSTRACT CLASS    
  def _rebuild_train_model (self) :
    self.model = ConditionedLSTMKerasModel (self.network_params, batch_size = None, lstm_dropout = self.lstm_dropout)
    self.model.compile (loss = self.loss_function, optimizer = self.optimizer)
    self.model.load_weights (tensorflow.train.latest_checkpoint (self.checkpoint_directory))
    self._is_in_generate_mode = False
        
 
  # THIS COULD BE FACTORED IN AN ABSTRACT CLASS    
  def train (self, nb_epochs = 1, nb_steps_per_epoch = 1, train_dataset = None, checkpoint_name_prefix = None) :
    if (checkpoint_name_prefix is None) :
      checkpoint_name_prefix = self.model_name
    if (self._is_in_generate_mode) :
      self._rebuild_train_model ()
    self.checkpoint_name_prefix = checkpoint_name_prefix
    self.model.compile (loss = self.loss_function, optimizer = self.optimizer)
    checkpoint_callback = tensorflow.keras.callbacks.ModelCheckpoint (filepath = os.path.join (self.checkpoint_directory, self.checkpoint_name_prefix),
                                                                            save_weights_only = True)
    history = self.model.fit (train_dataset.repeat (),
                                  epochs = nb_epochs,
                                  steps_per_epoch = nb_steps_per_epoch,
                                  callbacks = [ checkpoint_callback, ])
    return history
  
  def load_checkpoint_for_use (self, batch_size = 1) :
    self.model = ConditionedLSTMKerasModel (self.network_params, lstm_dropout= 0., batch_size = batch_size)
    self.model.load_weights (tensorflow.train.latest_checkpoint (self.checkpoint_directory))
    self._is_in_generate_mode = True
    
  def load_checkpoint_for_training (self, batch_size = None) :
    self._is_in_generate_mode = False
    self.model = ConditionedLSTMKerasModel (self.network_params, lstm_dropout= self.lstm_dropout, batch_size = batch_size)
    self.model.compile (loss = self.loss_function, optimizer = self.optimizer)
    self.model.load_weights (tensorflow.train.latest_checkpoint (self.checkpoint_directory))
  
  def save_model (self, file_name = None) :
    raise Exception ('this is not going to work')
    if (file_name is None) :
      file_name = self.model_name + '.h5'
    self.model.save (file_name)
    
  def load_model (self, file_name = None) :
    raise Exception ('this is not going to work')
    if (file_name is None) :
      file_name = self.model_name + '.h5'
    self.model = tensorflow.keras.models.load_model (file_name)
  
  # THIS COULD BE FACTORED IN AN ABSTRACT CLASS    
  def set_generate_mode (self) :
    if (self._is_in_generate_mode) :
      return
    self.model = ConditionedLSTMKerasModel (self.network_params, lstm_dropout= 0., batch_size = 1)
    self.model.load_weights (tensorflow.train.latest_checkpoint (self.checkpoint_directory))
    self._is_in_generate_mode = True
    
  # THIS COULD BE FACTORED IN AN ABSTRACT CLASS    
  def set_training_mode (self) :
    raise Exception ('TODO')

  # THIS COULD BE FACTORED IN AN ABSTRACT CLASS    
  def eval (self, eval_metrics) :
    raise Exception ('TODO')
    
  def generate(self, seed_tokens, categories, length, temperature = 1.) :
  
    seed_len = len (seed_tokens)
    total_len = seed_len + length
    generated_tokens = numpy.zeros (total_len, dtype = 'int32')
    generated_tokens [ : seed_len ] = seed_tokens
  
    categories_inputs = tensorflow.expand_dims(repeat_categories(categories, total_len), axis = 0)
    text_inputs = tensorflow.expand_dims(seed_tokens, 0)
    current_len = seed_len
    
    for i in range (length) :
      logits = self.model ((categories_inputs [ :, : current_len, : ], text_inputs)) [ : , -1 , : ]
      logits = logits / temperature
      generated_token = tensorflow.squeeze (tensorflow.random.categorical (logits, num_samples = 1)).numpy ()
      generated_tokens [current_len] = generated_token
      current_len += 1
      text_inputs = tensorflow.expand_dims (generated_tokens [ : current_len ], 0)
    
    return generated_tokens


  def save_weights (self, path) :
    self.model.save_weights (path)

  def load_weights (self, path) :
    self.model.load_weights (path)












