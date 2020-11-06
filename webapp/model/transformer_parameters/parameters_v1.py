
version_name = 'v1'
do_compat_v6 = False

# model

# the tokenizer is the same as the v7 tokenizer
tokenizer_file_name = f'inscriptiones-tokenizer-v7'


is_char_level = False
tokenizer_type = 'house'

seq_max_len = 60

nb_layers = 2
nb_heads = 2
head_dim = 4
d_ff = 4
dropout_rate = 0.1


d_model = nb_heads * head_dim



# training

nb_epochs = 13
save_every = 13
batch_size = 64

checkpoint_path = f'{version_name}-{nb_layers}'



# not really important

dataset_buffer_size = 20000








