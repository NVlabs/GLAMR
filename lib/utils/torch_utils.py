# Some functions in this script are borrowed and extended from https://github.com/Khrylx/AgentFormer/blob/main/utils/torch.py
# Adhere to their licence to use this script


import torch
import numpy as np
from torch import nn
from torch.optim import lr_scheduler


class ExtModuleWrapper:

    def __init__(self, module):
        self.module = module

    def __call__(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def inference(self, *args, **kwargs):
        return self.module.inference(*args, **kwargs)

    def to(self, *args, **kwargs):
        return self.module.to(*args, **kwargs)

    def eval(self):
        return self.module.eval()


class to_cpu:

    def __init__(self, *models):
        self.models = list(filter(lambda x: x is not None, models))
        self.prev_devices = [x.device if hasattr(x, 'device') else next(x.parameters()).device for x in self.models]
        for x in self.models:
            x.to(torch.device('cpu'))

    def __enter__(self):
        pass

    def __exit__(self, *args):
        for x, device in zip(self.models, self.prev_devices):
            x.to(device)
        return False


class to_device:

    def __init__(self, device, *models):
        self.models = list(filter(lambda x: x is not None, models))
        self.prev_devices = [x.device if hasattr(x, 'device') else next(x.parameters()).device for x in self.models]
        for x in self.models:
            x.to(device)

    def __enter__(self):
        pass

    def __exit__(self, *args):
        for x, device in zip(self.models, self.prev_devices):
            x.to(device)
        return False


class to_test:

    def __init__(self, *models):
        self.models = list(filter(lambda x: x is not None, models))
        self.prev_modes = [x.training for x in self.models]
        for x in self.models:
            x.train(False)

    def __enter__(self):
        pass

    def __exit__(self, *args):
        for x, mode in zip(self.models, self.prev_modes):
            x.train(mode)
        return False


class to_train:

    def __init__(self, *models):
        self.models = list(filter(lambda x: x is not None, models))
        self.prev_modes = [x.training for x in self.models]
        for x in self.models:
            x.train(True)

    def __enter__(self):
        pass

    def __exit__(self, *args):
        for x, mode in zip(self.models, self.prev_modes):
            x.train(mode)
        return False


def batch_to(dst, *args):
    return [x.to(dst) if x is not None else None for x in args]


def tensor_to(args, device=None, dtype=None):
    if isinstance(args, torch.Tensor):
        args_new = args
        if device is not None:
            args_new = args_new.to(device)
        if dtype is not None:
            args_new = args_new.to(dtype)
        return args_new
    elif isinstance(args, np.ndarray):
        return tensor_to(torch.tensor(args), device, dtype)
    elif isinstance(args, list):
        return [tensor_to(x, device, dtype) for x in args]
    elif isinstance(args, dict):
        return {k: tensor_to(x, device, dtype) for k, x in args.items()}
    else:
        return args


def tensor_to_numpy(args):
    if isinstance(args, torch.Tensor):
        return args.detach().cpu().numpy()
    elif isinstance(args, list):
        return [tensor_to_numpy(x) for x in args]
    elif isinstance(args, dict):
        return {k: tensor_to_numpy(x) for k, x in args.items()}
    else:
        return args


def initialize_weights(modules):
    for m in modules:
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            if m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            if m.bias is not None: nn.init.constant_(m.bias, 0)