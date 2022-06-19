# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

import torch
import numpy as np

import torch.nn.functional as F


def perspective_projection(p3d, K, t_form=None):
    p2d = torch.matmul(K, p3d.transpose(2, 1)).transpose(2, 1)
    p2d = p2d[:, :, :2] / (p2d[:, :, 2:] + 1e-8)

    if t_form is not None:
        ones = torch.ones((p2d.shape[0], p2d.shape[1], 1)).cuda().float()
        p2d = torch.cat((p2d, ones), dim=2)
        p2d = torch.matmul(p2d, t_form.transpose(2,1))
        p2d = p2d[:,:,:2]
    return p2d
