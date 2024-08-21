import time
import torch
import torch.nn as nn
from tqdm import tqdm
from timm.utils import AverageMeter

from openstl.models import SimVP_Model
from .base_method import Base_method
from openstl.utils import (reduce_tensor, reshape_patch, reshape_patch_back)
import math

class SimVP(Base_method):
    r"""SimVP

    Implementation of `SimVP: Simpler yet Better Video Prediction
    <https://arxiv.org/abs/2206.05099>`_.

    """

    def __init__(self, args, device, steps_per_epoch):
        Base_method.__init__(self, args, device, steps_per_epoch)
        self.model = self._build_model(self.config)
        self.model_optim, self.scheduler, self.by_epoch = self._init_optimizer(steps_per_epoch)
        self.criterion = nn.MSELoss()

    def _build_model(self, args):
        return SimVP_Model(**args).to(self.device)

    def _predict(self, batch_x, batch_y=None, **kwargs):
        """Forward the model"""
        img_size,_ = self.args.in_shape
        if self.args.fusion_method == 'mix':
            cell_length = int(math.sqrt(self.args.total_channel))
            batch_x_copy = batch_x.clone().permute(0, 1, 3, 4, 2).contiguous()
            batch_x_copy = reshape_patch(batch_x_copy, img_size)
            batch_x_copy = batch_x_copy.view(-1, self.args.total_length, img_size*cell_length, img_size//cell_length, cell_length, cell_length).transpose(3,4).contiguous()
            batch_x_copy = batch_x_copy.view(-1, self.args.total_length, self.args.total_channel, img_size, img_size)
            batch_x_copy = torch.cat([batch_x[:,:,:self.args.img_channel], batch_x_copy], dim=2)
        
        if self.args.aft_seq_length == self.args.pre_seq_length:
            if self.args.fusion_method == 'mix':
                pred_y = self.model(batch_x_copy)
            else:
                pred_y = self.model(batch_x)
        elif self.args.aft_seq_length < self.args.pre_seq_length:
            if self.args.fusion_method == 'mix':
                pred_y = self.model(batch_x_copy)
            else:
                pred_y = self.model(batch_x)
            pred_y = pred_y[:, :self.args.aft_seq_length]
        
        if self.args.aft_seq_length > self.args.pre_seq_length:
            list = []
            d = self.args.aft_seq_length // self.args.pre_seq_length
            m = self.args.aft_seq_length % self.args.pre_seq_length
            
            cur_seq = batch_x[:, :self.args.pre_seq_length].clone()
            #cur_seq = cur_seq[:, :, self.args.img_channel:]
            for i in range(d):
                if self.args.aux_channel == 0:
                    pred_y = self.model(cur_seq)
                    cur_seq = pred_y
                elif self.args.aux_channel > 0:
                    if self.args.fusion_method == 'mix':
                        cur_seq = cur_seq.permute(0, 1, 3, 4, 2).contiguous()
                        cur_seq = reshape_patch(cur_seq, img_size)
                        cur_seq = cur_seq.view(-1, self.args.pre_seq_length, img_size*cell_length, img_size//cell_length, cell_length, cell_length).transpose(3,4).contiguous()
                        cur_seq = cur_seq.view(-1, self.args.pre_seq_length, self.args.total_channel, img_size, img_size)
                        if i == 0:
                            cur_seq = torch.cat([batch_x[:,(self.args.pre_seq_length*i):(self.args.pre_seq_length*(i+1)),:self.args.img_channel],cur_seq], dim=2)
                        else:
                            cur_seq = torch.cat([list[-1], cur_seq], dim=2)
                        pred_y = self.model(cur_seq)
                        cur_seq = torch.cat([pred_y,batch_x[:,(self.args.pre_seq_length*(i+1)):(self.args.pre_seq_length*(i+2)),self.args.img_channel:]], dim=2)
                        #cur_seq = batch_x[:,(self.args.pre_seq_length*(i+1)):(self.args.pre_seq_length*(i+2)),self.args.img_channel:]
                    else:
                        pred_y = self.model(cur_seq)
                        cur_seq = torch.cat([pred_y,batch_x[:,(self.args.pre_seq_length*(i+1)):(self.args.pre_seq_length*(i+2)),self.args.img_channel:]], dim=2)
                list.append(pred_y)

            if m != 0:
                if self.args.fusion_method == 'mix':
                    cur_seq = cur_seq.permute(0, 1, 3, 4, 2).contiguous()
                    cur_seq = reshape_patch(cur_seq, img_size)
                    cur_seq = cur_seq.view(-1, self.args.pre_seq_length, img_size*cell_length, img_size//cell_length, cell_length, cell_length).transpose(3,4).contiguous()
                    cur_seq = cur_seq.view(-1, self.args.pre_seq_length, self.args.total_channel, img_size, img_size)
                    cur_seq = torch.cat([list[-1], cur_seq], dim=2)
                    pred_y = self.model(cur_seq)
                else:
                    cur_seq = self.model(cur_seq)
                list.append(cur_seq[:, :m])
            
            pred_y = torch.cat(list, dim=1)
            '''
            pred_y = pred_y.view(-1, self.args.aft_seq_length, self.args.img_channel, img_size * img_size)
            #pred_y, _ = torch.sort(pred_y, dim=3)
            _, pred_y = torch.sort(pred_y, dim=3)
            _, pred_y = torch.sort(pred_y, dim=3)
            pred_y = pred_y.to(dtype=torch.float32)/(img_size*img_size)
            pred_y = pred_y.view(-1, self.args.aft_seq_length, self.args.img_channel, img_size, img_size)
            '''
        return pred_y

    def train_one_epoch(self, runner, train_loader, epoch, num_updates, eta=None, **kwargs):
        """Train the model with train_loader."""
        data_time_m = AverageMeter()
        losses_m = AverageMeter()
        self.model.train()
        if self.by_epoch:
            self.scheduler.step(epoch)
        train_pbar = tqdm(train_loader) if self.rank == 0 else train_loader

        end = time.time()
        for batch_x, batch_y in train_pbar:
            data_time_m.update(time.time() - end)
            self.model_optim.zero_grad()

            if not self.args.use_prefetcher:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
            runner.call_hook('before_train_iter')

            with self.amp_autocast():
                pred_y = self._predict(batch_x)
                loss = self.criterion(pred_y, batch_y)

            if not self.dist:
                losses_m.update(loss.item(), batch_x.size(0))

            if self.loss_scaler is not None:
                if torch.any(torch.isnan(loss)) or torch.any(torch.isinf(loss)):
                    raise ValueError("Inf or nan loss value. Please use fp32 training!")
                self.loss_scaler(
                    loss, self.model_optim,
                    clip_grad=self.args.clip_grad, clip_mode=self.args.clip_mode,
                    parameters=self.model.parameters())
            else:
                loss.backward()
                self.clip_grads(self.model.parameters())
                self.model_optim.step()

            torch.cuda.synchronize()
            num_updates += 1

            if self.dist:
                losses_m.update(reduce_tensor(loss), batch_x.size(0))

            if not self.by_epoch:
                self.scheduler.step()
            runner.call_hook('after_train_iter')
            runner._iter += 1

            if self.rank == 0:
                log_buffer = 'train loss: {:.4f}'.format(loss.item())
                log_buffer += ' | data time: {:.4f}'.format(data_time_m.avg)
                train_pbar.set_description(log_buffer)

            end = time.time()  # end for

        if hasattr(self.model_optim, 'sync_lookahead'):
            self.model_optim.sync_lookahead()

        return num_updates, losses_m, eta
