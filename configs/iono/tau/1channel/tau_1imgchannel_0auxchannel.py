method = 'TAU'
# dataset
img_channel = 1
aux_channel = 0
total_channel = 1
fusion_method = None
# model
spatio_kernel_enc = 3
spatio_kernel_dec = 3
model_type = 'tau'
hid_S = 64
hid_T = 512
N_T = 8
N_S = 4
alpha = 0.1
# training
lr = 1e-3
batch_size = 8
val_batch_size = 4
epoch = 10
drop_path = 0
sched = 'onecycle'
