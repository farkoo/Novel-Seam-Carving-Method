from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import cv2
import numpy as np
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
import re
import sys
import os
import os.path as osp
from scipy.io import savemat
from GM_src import bdcn
from SM_src.model import SODModel
from SM_src.dataloader import InfDataloader, SODLoader

def ReturnToOriginSize(img_path, mask):
    original_image = cv2.imread(img_path);
    h = original_image.shape[0];
    w = original_image.shape[1];
    max_dim = max(h, w);
    temp_h = 0
    temp_w = 0
    if (max_dim - h) % 2 == 1:
      temp_h = 1
    if (max_dim - w) % 2 == 1:
      temp_w = 1
    resized_image = cv2.resize(mask,(max_dim - temp_w, max_dim - temp_h), interpolation=cv2.INTER_AREA);
    padding_h = (resized_image.shape[0] - h) // 2;
    padding_w = (resized_image.shape[1] - w) // 2;
    out_image = resized_image[padding_h:resized_image.shape[0]-padding_h, padding_w:resized_image.shape[1]-padding_w];
    return out_image;

def GM_creator(img_path, model_path, cuda):
    model = bdcn.BDCN()
    model.load_state_dict(torch.load(model_path))
    mean_bgr = np.array([104.00699, 116.66877, 122.67892])
    if cuda:
      model.cuda()
    model.eval()
    data = cv2.imread(img_path)
    data = np.array(data, np.float32)
    data -= mean_bgr
    data = data.transpose((2, 0, 1))
    data = torch.from_numpy(data).float().unsqueeze(0)
    if cuda:
      data = data.cuda()
    data = Variable(data)
    out = model(data)
    out = [torch.sigmoid(out[-1]).cpu().data.numpy()[0, 0, :, :]]
    gradient_map = out[-1]
    return gradient_map

def SM_creator(img_path, model_path, img_size, bs, device):
    model = SODModel()
    chkpt = torch.load(model_path, map_location=device)
    model.load_state_dict(chkpt['model'])
    model.to(device)
    model.eval()
    inf_data = InfDataloader(img_path=img_path, target_size=256)
    inf_dataloader = DataLoader(inf_data, batch_size=1, shuffle=True, num_workers=2)
    with torch.no_grad():
      for batch_idx, (img_np, img_tor) in enumerate(inf_dataloader, start=1):
        img_tor = img_tor.to(device)
        pred_masks, _ = model(img_tor)
        img_np = np.squeeze(img_np.numpy(), axis=0)
        img_np = img_np.astype(np.uint8)
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        mask = np.squeeze(pred_masks.cpu().numpy(), axis=(0, 1))
        saliency_map = ReturnToOriginSize(img_path, mask)
    return saliency_map
      
def DM_creator(img_path):
    depth_map = cv2.imread(img_path)
    depth_map = cv2.cvtColor(depth_map, cv2.COLOR_RGB2GRAY)
    depth_map = depth_map/255.0
    return depth_map