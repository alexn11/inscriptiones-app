
lstm_models = [
  'lstm-char',
  'lstm-word',
  'lstm-char-cond',
  'lstm-word-cond'
]

#transformer_models_no_pretrain = [ 'transformer-v8', 'transformer-v9', 'transformer-v10', 'transformer-v11', 'transformer-v12', 'transformer-v13' ]
#transformer_models_pretrain = [ 'transformer-v15', 'transformer-v16', 'transformer-v17', ]
transformer_models_no_pretrain = [ 'transformer-v1', 'transformer-v11', 'transformer-v20' ]
transformer_models_pretrain = []

transformer_models = transformer_models_no_pretrain + transformer_models_pretrain

model_list = lstm_models + transformer_models + [ 'bunny', ]

model_uses_categories = {
 'lstm-char' : False,
 'lstm-word' : False,
 'lstm-char-cond' : True,
 'lstm-word-cond' : True,
 'transformer-v1' : False,
# 'transformer-v8' : False,
# 'transformer-v9' : False,
# 'transformer-v10' : False,
 'transformer-v11' : False,
# 'transformer-v12' : False,
# 'transformer-v13' : False,
# 'transformer-v15' : False,
# 'transformer-v16' : False,
# 'transformer-v17' : False,
 'transformer-v20' : False,
}
