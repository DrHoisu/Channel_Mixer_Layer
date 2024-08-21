method = 'FMN'
# dataset
img_channel = 1
aux_channel = 0
total_channel = 1
# reverse scheduled sampling
reverse_scheduled_sampling = 0
r_sampling_step_1 = 25000
r_sampling_step_2 = 50000
r_exp_alpha = 5000
# scheduled sampling
scheduled_sampling = 1
sampling_stop_iter = 50000
sampling_start_value = 1.0
sampling_changing_rate = 0.00002
# model
method_type = 'mim'
exp_stage = 'test_whole_net'
value_net_sort = 'sort_after_pred'
index_integerize = 'index_round'
num_hidden = '128,128,128,128'
filter_size = 5
stride = 1
patch_size = 4
layer_norm = 0
# training
lr = 5e-4
epoch = 200
batch_size = 16
sched = 'onecycle'

# EMAHook = dict(
#     momentum=0.999,
#     priority='NORMAL',
# )
