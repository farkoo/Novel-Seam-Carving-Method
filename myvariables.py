import torch

_orgimg_path = 'images/art-v.png'
_dispimg_path = 'depth_map/art-d.png'
orgimg_path = 'img_temp/img_temp.png'
dispimg_path = 'disp_temp/disp_temp.png'
gmodel_path = 'final-models/bdcn_bsds500.pth'
smodel_path = 'final-models/best-model.pth'
img_size = 256
bs = 24
cuda = True
device = torch.device(device='cuda')
# cuda = False
# device = torch.device(device='cpu')