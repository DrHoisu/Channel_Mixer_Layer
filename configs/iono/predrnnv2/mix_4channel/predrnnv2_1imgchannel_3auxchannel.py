method = 'PredRNNv2'
# dataset
img_channel = 1
aux_channel = 3
total_channel = 4
fusion_method = 'mix'
# reverse scheduled sampling
reverse_scheduled_sampling = 1
r_sampling_step_1 = 25000
r_sampling_step_2 = 50000
r_exp_alpha = 5000
# scheduled sampling
scheduled_sampling = 1
sampling_stop_iter = 50000
sampling_start_value = 1.0
sampling_changing_rate = 0.00002
# model
num_hidden = '128,128,128,128'
filter_size = 5
stride = 1
patch_size = 4
layer_norm = 0
decouple_beta = 0.1
# training
lr = 5e-4
batch_size = 8
val_batch_size = 4
epoch = 10
sched = 'onecycle'
