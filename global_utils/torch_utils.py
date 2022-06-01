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


def set_optimizer_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def filter_state_dict(state_dict, filter_keys):
    for key in list(state_dict.keys()):
        for f_key in filter_keys:
            if f_key in key:
                del state_dict[key]
                break


def get_scheduler(optimizer, policy, nepoch_fix=None, nepoch=None, decay_step=None):
    if policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch - nepoch_fix) / float(nepoch - nepoch_fix + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif policy == 'step':
        scheduler = lr_scheduler.StepLR(
            optimizer, step_size=decay_step, gamma=0.1)
    elif policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', policy)
    return scheduler


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