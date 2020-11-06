
version_name = 'v20' # from v12
do_compat_v6 = False

# model

# the tokenizer is the same as the v7 tokenizer
tokenizer_file_name = f'inscriptiones-tokenizer-v7'



is_char_level = False
tokenizer_type = 'house'

seq_max_len = 60

nb_layers = 6 # 14 is the max wrt OOM
nb_heads = 24
head_dim = 64
d_ff = 512
dropout_rate = 0.1

# nlayer head dim: 6x64 -> OOM try maybe 8x32?
# also nb of heads

d_model = nb_heads * head_dim



# training

checkpoint_path = f'{version_name}-{nb_layers}'

nb_pretrain_epochs = 12
nb_finetune_epochs = 4
nb_epochs = 20
save_every = 1
batch_size = 64


# not really important

dataset_buffer_size = 20000








