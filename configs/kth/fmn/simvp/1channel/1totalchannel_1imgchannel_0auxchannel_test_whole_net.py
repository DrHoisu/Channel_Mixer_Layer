method = 'FMN'
# dataset
img_channel = 1
aux_channel = 0
total_channel = 1
# model
spatio_kernel_enc = 3
spatio_kernel_dec = 3
model_type = 'gsta'
method_type = 'simvp'
exp_stage = 'test_whole_net'
value_net_sort = 'sort_before_pred'
index_integerize = 'index_round'
hid_S = 64
hid_T = 256
N_T = 6
N_S = 2
# training
lr = 1e-3
epoch = 100
batch_size = 16
drop_path = 0.1
sched = 'onecycle'

# EMAHook = dict(
#     momentum=0.999,
#     priority='NORMAL',
# )
