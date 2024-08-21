import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from openstl.modules import SpatioTemporalLSTMCellv2
from openstl.utils import reshape_patch

class PredRNNv2_Model(nn.Module):
    r"""PredRNNv2 Model

    Implementation of `PredRNN: A Recurrent Neural Network for Spatiotemporal
    Predictive Learning <https://arxiv.org/abs/2103.09504v4>`_.

    """

    def __init__(self, num_layers, num_hidden, configs, **kwargs):
        super(PredRNNv2_Model, self).__init__()
        total_channel = configs.total_channel + configs.img_channel if configs.fusion_method == 'mix' else configs.total_channel
        T, C = configs.pre_seq_length, total_channel
        H, W = configs.in_shape
        self.img_channel, self.aux_channel, self.fusion_method = configs.img_channel, configs.aux_channel, configs.fusion_method

        self.configs = configs

        self.frame_channel = configs.patch_size * configs.patch_size * C
        self.pred_channel = configs.patch_size * configs.patch_size * configs.img_channel
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        cell_list = []

        height = H // configs.patch_size
        width = W // configs.patch_size
        self.MSE_criterion = nn.MSELoss()

        for i in range(num_layers):
            in_channel = self.frame_channel if i == 0 else num_hidden[i - 1]
            cell_list.append(
                SpatioTemporalLSTMCellv2(in_channel, num_hidden[i], height, width,
                                         configs.filter_size, configs.stride, configs.layer_norm))
        self.cell_list = nn.ModuleList(cell_list)
        self.conv_last = nn.Conv2d(num_hidden[num_layers - 1], self.pred_channel, kernel_size=1,
                                   stride=1, padding=0, bias=False)
        # shared adapter
        adapter_num_hidden = num_hidden[0]
        self.adapter = nn.Conv2d(
            adapter_num_hidden, adapter_num_hidden, 1, stride=1, padding=0, bias=False)

    def forward(self, frames_tensor, mask_true, **kwargs):
        return_loss = kwargs.get('return_loss', True)
        # [batch, length, height, width, channel] -> [batch, length, channel, height, width]
        frames = frames_tensor.permute(0, 1, 4, 2, 3).contiguous()
        mask_true = mask_true.permute(0, 1, 4, 2, 3).contiguous()

        batch = frames.shape[0]
        height = frames.shape[3]
        width = frames.shape[4]

        next_frames = []
        h_t = []
        c_t = []
        delta_c_list = []
        delta_m_list = []

        decouple_loss = []

        aux_frames = frames.clone()
        if self.configs.fusion_method == 'mix':
            #with torch.no_grad():
            cell_length = int(math.sqrt(self.configs.total_channel))
            frames_copy = reshape_patch(frames_tensor, width)
            frames_copy = frames_copy.view(-1, self.configs.total_length, width*cell_length, width//cell_length, cell_length, cell_length).transpose(3,4).contiguous()
            frames_copy = frames_copy.view(-1, self.configs.total_length, self.configs.total_channel*self.configs.patch_size **2, width, width)
            frames = torch.cat([frames[:,:,::(self.img_channel+self.aux_channel)], frames_copy], dim=2)

        for i in range(self.num_layers):
            zeros = torch.zeros(
                [batch, self.num_hidden[i], height, width]).to(self.configs.device)
            h_t.append(zeros)
            c_t.append(zeros)
            delta_c_list.append(zeros)
            delta_m_list.append(zeros)

        memory = torch.zeros(
            [batch, self.num_hidden[0], height, width]).to(self.configs.device)

        for t in range(self.configs.total_length - 1):

            if self.configs.reverse_scheduled_sampling == 1:
                # reverse schedule sampling
                if t == 0:
                    net = frames[:, t]
                else:
                    net = mask_true[:, t - 1] * frames[:, t] + (1 - mask_true[:, t - 1]) * x_gen
            else:
                # schedule sampling
                if t < self.configs.pre_seq_length:
                    net = frames[:, t]
                else:
                    net = mask_true[:, t - self.configs.pre_seq_length] * frames[:, t] + \
                          (1 - mask_true[:, t - self.configs.pre_seq_length]) * x_gen

            h_t[0], c_t[0], memory, delta_c, delta_m = \
                self.cell_list[0](net, h_t[0], c_t[0], memory)
            delta_c_list[0] = F.normalize(
                self.adapter(delta_c).view(delta_c.shape[0], delta_c.shape[1], -1), dim=2)
            delta_m_list[0] = F.normalize(
                self.adapter(delta_m).view(delta_m.shape[0], delta_m.shape[1], -1), dim=2)

            for i in range(1, self.num_layers):
                h_t[i], c_t[i], memory, delta_c, delta_m = \
                    self.cell_list[i](h_t[i - 1], h_t[i], c_t[i], memory)
                delta_c_list[i] = F.normalize(
                    self.adapter(delta_c).view(delta_c.shape[0], delta_c.shape[1], -1), dim=2)
                delta_m_list[i] = F.normalize(
                    self.adapter(delta_m).view(delta_m.shape[0], delta_m.shape[1], -1), dim=2)

            gen_frames = self.conv_last(h_t[self.num_layers - 1])
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
                    x_gen = reshape_patch(x_gen, width)
                    x_gen = x_gen.view(-1, width*cell_length, width//cell_length, cell_length, cell_length).transpose(2,3).contiguous()
                    x_gen = x_gen.view(-1, self.configs.total_channel*self.configs.patch_size **2, width, width)
                    x_gen = torch.cat([gen_frames, x_gen], dim=1)

            # decoupling loss
            if return_loss:
                for i in range(0, self.num_layers):
                    decouple_loss.append(torch.mean(torch.abs(
                        torch.cosine_similarity(delta_c_list[i], delta_m_list[i], dim=2))))

        if return_loss:
            decouple_loss = torch.mean(torch.stack(decouple_loss, dim=0))

        # [length, batch, channel, height, width] -> [batch, length, height, width, channel]
        next_frames = torch.stack(next_frames, dim=0).permute(1, 0, 3, 4, 2).contiguous()
        if return_loss:
            loss = self.MSE_criterion(next_frames, frames_tensor[:, 1:, :, :, ::(self.img_channel+self.aux_channel)]) + \
                self.configs.decouple_beta * decouple_loss
        else:
            loss = None

        return next_frames, loss
