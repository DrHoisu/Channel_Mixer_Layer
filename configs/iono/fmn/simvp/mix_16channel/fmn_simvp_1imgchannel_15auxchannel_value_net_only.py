method = 'FMN'
# dataset
img_channel = 1
aux_channel = 15
total_channel = 16
fusion_method = 'mix'
# model
spatio_kernel_enc = 3
spatio_kernel_dec = 3
model_type = 'gsta'
method_type = 'simvp'
exp_stage = 'value_net_only'
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
