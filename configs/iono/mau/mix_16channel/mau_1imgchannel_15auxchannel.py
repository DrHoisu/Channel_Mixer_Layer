method = 'MAU'
# dataset
img_channel = 1
aux_channel = 15
total_channel = 16
fusion_method = 'mix'
# scheduled sampling
scheduled_sampling = 1
sampling_stop_iter = 50000
sampling_start_value = 1.0
sampling_changing_rate = 0.00002
# model
num_hidden = '64,64,64,64'
filter_size = 5
stride = 1
patch_size = 1
layer_norm = 0
sr_size = 4
tau = 5
cell_mode = 'normal'
model_mode = 'normal'
# training
lr = 1e-3
batch_size = 8
val_batch_size = 4
epoch = 10
sched = 'onecycle'
