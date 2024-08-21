method = 'FMN'
#dataset
img_channel = 1
aux_channel = 0
total_channel = 1
fusion_method = None
# model
spatio_kernel_enc = 3
spatio_kernel_dec = 3
model_type = 'gsta'
method_type = 'simvp'
exp_stage = 'index_net_only'
value_net_sort = 'sort_after_pred'
index_integerize = 'index_round'
hid_S = 64
hid_T = 512
N_T = 8
N_S = 4
# training
lr = 1e-3
epoch = 10
batch_size = 8
val_batch_size = 4
drop_path = 0
sched = 'onecycle'

# EMAHook = dict(
#     momentum=0.999,
#     priority='NORMAL',
# )
