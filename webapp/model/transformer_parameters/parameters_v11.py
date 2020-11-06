
version_name = 'v11'
do_compat_v6 = False

# model

# the tokenizer is the same as the v7 tokenizer
tokenizer_file_name = f'inscriptiones-tokenizer-v7'


is_char_level = False
tokenizer_type = 'house'

seq_max_len = 60

nb_layers = 4 # 14 is the max wrt OOM
nb_heads = 16
head_dim = 32
d_ff = 512
dropout_rate = 0.1

# nlayer head dim: 6x64 -> OOM try maybe 8x32?
# also nb of heads

d_model = nb_heads * head_dim



# training

nb_epochs = 12
save_every = 12
batch_size = 64

#checkpoint_path = "./checkpoints/train-v2"
#checkpoint_path = f'./checkpoints/train-v3-NL-{nb_layers}'
#checkpoint_path = f'./checkpoints/v5-autoreg-{nb_layers}'
checkpoint_path = f'{version_name}-{nb_layers}'

train_history_file_name = f'{version_name}-train_hist.csv'
do_save_train_history = True


# not really important

dataset_buffer_size = 20000








