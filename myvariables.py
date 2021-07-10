import torch

_img_folder = 'images'
_orgimg_path = 'images/view1.png'
_dispimg_path = 'depth_map/disp1.png'
img_folder = 'img_temp'
orgimg_path = 'img_temp/img_temp.png'
dispimg_path = 'disp_temp/disp_temp.png'
gmodel_path = 'final-models/bdcn_bsds500.pth'
smodel_path = 'final-models/best-model.pth'
img_size = 256
bs = 24
cuda = True
device = torch.device(device='cuda')