import math
import torch
import torch.nn as nn
from openstl.utils import reshape_patch
from openstl.modules import MAUCell


class MAU_Model(nn.Module):
    r"""MAU Model

    Implementation of `MAU: A Motion-Aware Unit for Video Prediction and Beyond
    <https://openreview.net/forum?id=qwtfY-3ibt7>`_.

    """

    def __init__(self, num_layers, num_hidden, configs, **kwargs):
        super(MAU_Model, self).__init__()
        total_channel = configs.total_channel + configs.img_channel if configs.fusion_method == 'mix' else configs.total_channel
        T, C = configs.pre_seq_length, total_channel
        H, W = configs.in_shape
        self.img_channel, self.aux_channel, self.fusion_method = configs.img_channel, configs.aux_channel, configs.fusion_method
        
        self.configs = configs
        self.frame_channel = configs.patch_size * configs.patch_size * C
        self.pred_channel = configs.patch_size * configs.patch_size * configs.img_channel
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        self.tau = configs.tau
        self.cell_mode = configs.cell_mode
        self.states = ['recall', 'normal']
        if not self.configs.model_mode in self.states:
            raise AssertionError
        cell_list = []

        width = W // configs.patch_size // configs.sr_size
        height = H // configs.patch_size // configs.sr_size
        self.MSE_criterion = nn.MSELoss()

        for i in range(num_layers):
            in_channel = num_hidden[i - 1]
            cell_list.append(
                MAUCell(in_channel, num_hidden[i], height, width, configs.filter_size,
                        configs.stride, self.tau, self.cell_mode)
            )
        self.cell_list = nn.ModuleList(cell_list)

        # Encoder
        n = int(math.log2(configs.sr_size))
        encoders = []
        encoder = nn.Sequential()
        encoder.add_module(name='encoder_t_conv{0}'.format(-1),
                           module=nn.Conv2d(in_channels=self.frame_channel,
                                            out_channels=self.num_hidden[0],
                                            stride=1,
                                            padding=0,
                                            kernel_size=1))
        encoder.add_module(name='relu_t_{0}'.format(-1),
                           module=nn.LeakyReLU(0.2))
        encoders.append(encoder)
        for i in range(n):
            encoder = nn.Sequential()
            encoder.add_module(name='encoder_t{0}'.format(i),
                               module=nn.Conv2d(in_channels=self.num_hidden[0],
                                                out_channels=self.num_hidden[0],
                                                stride=(2, 2),
                                                padding=(1, 1),
                                                kernel_size=(3, 3)
                                                ))
            encoder.add_module(name='encoder_t_relu{0}'.format(i),
                               module=nn.LeakyReLU(0.2))
            encoders.append(encoder)
        self.encoders = nn.ModuleList(encoders)

        # Decoder
        decoders = []

        for i in range(n - 1):
            decoder = nn.Sequential()
            decoder.add_module(name='c_decoder{0}'.format(i),
                               module=nn.ConvTranspose2d(in_channels=self.num_hidden[-1],
                                                         out_channels=self.num_hidden[-1],
                                                         stride=(2, 2),
                                                         padding=(1, 1),
                                                         kernel_size=(3, 3),
                                                         output_padding=(1, 1)
                                                         ))
            decoder.add_module(name='c_decoder_relu{0}'.format(i),
                               module=nn.LeakyReLU(0.2))
            decoders.append(decoder)

        if n > 0:
            decoder = nn.Sequential()
            decoder.add_module(name='c_decoder{0}'.format(n - 1),
                               module=nn.ConvTranspose2d(in_channels=self.num_hidden[-1],
                                                         out_channels=self.pred_channel,
                                                         stride=(2, 2),
                                                         padding=(1, 1),
                                                         kernel_size=(3, 3),
                                                         output_padding=(1, 1)
                                                         ))
            decoders.append(decoder)
        self.decoders = nn.ModuleList(decoders)

        #self.srcnn = nn.Sequential(
        #    nn.Conv2d(self.num_hidden[-1], self.pred_channel, kernel_size=1, stride=1, padding=0)
        #)
        #self.merge = nn.Conv2d(
        #    self.num_hidden[-1] * 2, self.num_hidden[-1], kernel_size=1, stride=1, padding=0)
        #self.conv_last_sr = nn.Conv2d(
        #    self.frame_channel * 2, self.pred_channel, kernel_size=1, stride=1, padding=0)

    def forward(self, frames_tensor, mask_true, **kwargs):
        # [batch, length, height, width, channel] -> [batch, length, channel, height, width]
        frames = frames_tensor.permute(0, 1, 4, 2, 3).contiguous()
        mask_true = mask_true.permute(0, 1, 4, 2, 3).contiguous()

        batch_size = frames.shape[0]
        height = frames.shape[3] // self.configs.sr_size
        width = frames.shape[4] // self.configs.sr_size
        img_width = frames.shape[4]
        frame_channels = frames.shape[2]
        next_frames = []
        T_t = []
        T_pre = []
        S_pre = []
        x_gen = None

        aux_frames = frames.clone()
        if self.configs.fusion_method == 'mix':
            #with torch.no_grad():
            cell_length = int(math.sqrt(self.configs.total_channel))
            frames_copy = reshape_patch(frames_tensor, img_width)
            frames_copy = frames_copy.view(-1, self.configs.total_length, img_width*cell_length, img_width//cell_length, cell_length, cell_length).transpose(3,4).contiguous()
            frames_copy = frames_copy.view(-1, self.configs.total_length, self.configs.total_channel*self.configs.patch_size **2, img_width, img_width)
            #frames = torch.cat([frames[:,:,::(self.img_channel+self.aux_channel)], frames_copy], dim=2)
            frames = frames_copy

        for layer_idx in range(self.num_layers):
            tmp_t = []
            tmp_s = []
            if layer_idx == 0:
                in_channel = self.num_hidden[layer_idx]
            else:
                in_channel = self.num_hidden[layer_idx - 1]
            for i in range(self.tau):
                tmp_t.append(torch.zeros(
                    [batch_size, in_channel, height, width]).to(self.configs.device))
                tmp_s.append(torch.zeros(
                    [batch_size, in_channel, height, width]).to(self.configs.device))
            T_pre.append(tmp_t)
            S_pre.append(tmp_s)

        for t in range(self.configs.total_length - 1):
            if t < self.configs.pre_seq_length:
                net = frames[:, t]
            else:
                time_diff = t - self.configs.pre_seq_length
                net = mask_true[:, time_diff] * frames[:, t] + (1 - mask_true[:, time_diff]) * x_gen
            frames_feature = net
            frames_feature_encoded = []
            for i in range(len(self.encoders)):
                frames_feature = self.encoders[i](frames_feature)
                frames_feature_encoded.append(frames_feature)
            if t == 0:
                for i in range(self.num_layers):
                    zeros = torch.zeros(
                        [batch_size, self.num_hidden[i], height, width]).to(self.configs.device)
                    T_t.append(zeros)
            S_t = frames_feature
            for i in range(self.num_layers):
                t_att = T_pre[i][-self.tau:]
                t_att = torch.stack(t_att, dim=0)
                s_att = S_pre[i][-self.tau:]
                s_att = torch.stack(s_att, dim=0)
                S_pre[i].append(S_t)
                T_t[i], S_t = self.cell_list[i](T_t[i], S_t, t_att, s_att)
                T_pre[i].append(T_t[i])
            out = S_t

            for i in range(len(self.decoders)):
                out = self.decoders[i](out)
                if self.configs.model_mode == 'recall':
                    out = out + frames_feature_encoded[-2 - i]

            #gen_frames = self.srcnn(out)
            gen_frames = out
            next_frames.append(gen_frames)

            if self.aux_channel == 0:
                x_gen = gen_frames
            else:
                #with torch.no_grad():
                gen_frames_split = torch.chunk(gen_frames, self.configs.patch_size **2, dim=1)
                aux_frames_split = torch.chunk(aux_frames[:, t+1], self.configs.patch_size **2, dim=1)
                length = len(gen_frames_split)
                split_list = []
                for i in range(length):
                    split_list.append(gen_frames_split[i])
                    split_list.append(aux_frames_split[i][:, self.img_channel:])
                x_gen = torch.cat(split_list, dim=1)

                if self.fusion_method == 'mix':
                    x_gen = x_gen.permute(0, 2, 3, 1).contiguous()
                    x_gen = reshape_patch(x_gen, img_width)
                    x_gen = x_gen.view(-1, img_width*cell_length, img_width//cell_length, cell_length, cell_length).transpose(2,3).contiguous()
                    x_gen = x_gen.view(-1, self.configs.total_channel*self.configs.patch_size **2, img_width, img_width)
                    #x_gen = torch.cat([gen_frames, x_gen], dim=1)

        # [length, batch, channel, height, width] -> [batch, length, height, width, channel]
        next_frames = torch.stack(next_frames, dim=0).permute(1, 0, 2, 3, 4).contiguous()
        if kwargs.get('return_loss', True):
            loss = self.MSE_criterion(next_frames, frames[:, 1:, ::(self.img_channel+self.aux_channel), :, :])
        else:
            loss = None

        return next_frames, loss
