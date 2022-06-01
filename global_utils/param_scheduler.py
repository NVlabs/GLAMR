import torch
from torch import nn


class StepParamScheduler(nn.Module):

    def __init__(self, start_val, step_size, gamma, smooth=False, cur_epoch=0):
        super().__init__()
        self.register_buffer('start_val', torch.tensor(start_val))
        self.register_buffer('step_size', torch.tensor(step_size))
        self.register_buffer('gamma', torch.tensor(gamma))
        self.register_buffer('cur_epoch', torch.tensor(cur_epoch))
        self.smooth = smooth

    def step(self):
        self.cur_epoch += 1

    def set_epoch(self, epoch):
        self.cur_epoch.fill_(epoch)

    def val(self):
        return self.start_val * self.gamma ** (self.cur_epoch / self.step_size if self.smooth else self.cur_epoch // self.step_size)


class LinearParamScheduler(nn.Module):

    def __init__(self, start_val, end_val, start_epoch=0, end_epoch=0, cur_epoch=0):
        super().__init__()
        self.register_buffer('start_val', torch.tensor(start_val))
        self.register_buffer('end_val', torch.tensor(end_val))
        self.register_buffer('start_epoch', torch.tensor(start_epoch))
        self.register_buffer('end_epoch', torch.tensor(end_epoch))
        self.register_buffer('cur_epoch', torch.tensor(cur_epoch))

    def step(self):
        self.cur_epoch += 1

    def set_epoch(self, epoch):
        self.cur_epoch.fill_(epoch)

    def val(self):
        return self.start_val + ((self.cur_epoch - self.start_epoch) / (self.end_epoch - self.start_epoch)).clamp(0.0, 1.0) * (self.end_val - self.start_val)