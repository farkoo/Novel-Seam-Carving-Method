import scipy
import numpy as np
from skimage import img_as_float
import cv2
from pylab import *
import myvariables
from map_generation_functions import ReturnToOriginSize, GM_creator, SM_creator, DM_creator
import os
from SM_src.dataloader import InfDataloader, SODLoader
from GM_src import bdcn
from SM_src.model import SODModel
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch

def find_seam(img, energy):
  minval = 1000
  minIndex = 0
  rows = energy.shape[0]
  columns = energy.shape[1]
  sOfIJ = np.zeros(shape=(rows, columns))
  np.copyto(sOfIJ, energy)

  for i in range(1, rows):
    for j in range(1, columns - 1):
      if j == 1:
        sOfIJ[i, j] = sOfIJ[i, j] + \
          min(sOfIJ[i - 1, j], sOfIJ[i - 1, j + 1])
      elif j == columns - 2:
        sOfIJ[i, j] = sOfIJ[i, j] + \
          min(sOfIJ[i - 1, j - 1], sOfIJ[i - 1, j])
      else:
        sOfIJ[i, j] = sOfIJ[i, j] + min(sOfIJ[i - 1, j - 1], sOfIJ[i- 1, j], sOfIJ[i - 1, j + 1])

  lastRow = sOfIJ[rows - 1, :]
  for p in range(1, columns - 1):
    if lastRow[p] < minval:
      minval = lastRow[p]
      minIndex = p

  return minval, minIndex, sOfIJ

def remove_seam(img, minIndex, sOfIJ):
  rows = img.shape[0]
  columns = img.shape[1]
  if len(img.shape) == 2:
    img = cv2.merge((img, img, img))
  removed_matrix = np.zeros(shape=(rows, columns - 1, 3))
  k = minIndex
  for i in range(rows - 1, -1, -1):
    b = img[i, :, :]  
    removed_matrix[i, :, :] = np.delete(b, k, axis=0)
    if i != 0:
      if k == 1:
        if sOfIJ[i - 1, k + 1] < sOfIJ[i - 1, k]:
          k = k + 1
      elif k == columns - 2:
        if sOfIJ[i - 1, k - 1] < sOfIJ[i - 1, k]:
          k = k - 1
      else:
        if sOfIJ[i - 1, k - 1] < sOfIJ[i - 1, k] and sOfIJ[i - 1, k - 1] < sOfIJ[i - 1, k + 1]:
          k = k - 1
        elif sOfIJ[i - 1, k + 1] < sOfIJ[i - 1, k] and sOfIJ[i - 1, k + 1] < sOfIJ[i - 1, k - 1]:
          k = k + 1
  return removed_matrix

def produce_emap():
  gradient_map = GM_creator(myvariables.orgimg_path, myvariables.gmodel_path, myvariables.cuda)
  saliency_map = SM_creator(myvariables.orgimg_path, myvariables.smodel_path, myvariables.img_size, myvariables.bs, myvariables.device)
  depth_map = DM_creator(myvariables.dispimg_path)
  g_importance = gradient_map.sum()/(gradient_map.shape[0]*gradient_map.shape[1])
  s_importance = saliency_map.sum()/(saliency_map.shape[0]*saliency_map.shape[1])
  d_importance = depth_map.sum()/(depth_map.shape[0]*depth_map.shape[1])
  g_coe = 4*g_importance
  s_coe = 2.5*s_importance
  d_coe = 1*d_importance
  coe_emap = (g_coe*gradient_map + s_coe*saliency_map + d_coe*depth_map)/(g_coe + s_coe + d_coe)
  return coe_emap

def find_energy_range(number):
  os.system('cp images/view1.png img_temp/img_temp.png')
  os.system('cp depth_map/disp1.png disp_temp/disp_temp.png')
  energy_list3 = []
  gmodel = bdcn.BDCN()
  gmodel.load_state_dict(torch.load(myvariables.gmodel_path, map_location=myvariables.device))
  mean_bgr = np.array([104.00699, 116.66877, 122.67892])
  if myvariables.cuda:
    gmodel.cuda()
  gmodel.eval()
  data = cv2.imread(myvariables._orgimg_path)
  data = np.array(data, np.float32)
  data -= mean_bgr
  data = data.transpose((2, 0, 1))
  data = torch.from_numpy(data).float().unsqueeze(0)
  if myvariables.cuda:
    data = data.cuda()
  data = Variable(data)
  out = gmodel(data)
  out = [torch.sigmoid(out[-1]).cpu().data.numpy()[0, 0, :, :]]
  gradient_map = out[-1]

  smodel = SODModel()
  chkpt = torch.load(myvariables.smodel_path, map_location=myvariables.device)
  smodel.load_state_dict(chkpt['model'])
  smodel.to(myvariables.device)
  smodel.eval()
  inf_data = InfDataloader(img_path=myvariables._orgimg_path, target_size=256)
  inf_dataloader = DataLoader(inf_data, batch_size=1, shuffle=True, num_workers=2)
  with torch.no_grad():
    for batch_idx, (img_np, img_tor) in enumerate(inf_dataloader, start=1):
      img_tor = img_tor.to(myvariables.device)
      pred_masks, _ = smodel(img_tor)
      img_np = np.squeeze(img_np.numpy(), axis=0)
      img_np = img_np.astype(np.uint8)
      img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
      mask = np.squeeze(pred_masks.cpu().numpy(), axis=(0, 1))
      saliency_map = ReturnToOriginSize(myvariables._orgimg_path, mask)

  depth_map = cv2.imread(myvariables._orgimg_path)
  depth_map = cv2.cvtColor(depth_map, cv2.COLOR_RGB2GRAY)
  depth_map = depth_map/255.0

  g_importance = gradient_map.sum()/(gradient_map.shape[0]*gradient_map.shape[1])
  s_importance = saliency_map.sum()/(saliency_map.shape[0]*saliency_map.shape[1])
  d_importance = depth_map.sum()/(depth_map.shape[0]*depth_map.shape[1])
  g_coe = 4*g_importance
  s_coe = 2.5*s_importance
  d_coe = 1*d_importance
  coe_emap = (g_coe*gradient_map + s_coe*saliency_map + d_coe*depth_map)/(g_coe + s_coe + d_coe)

  emap = coe_emap
  reduced_image3 = cv2.imread(myvariables._orgimg_path)
  disp = cv2.imread(myvariables._dispimg_path)
  for i in range(number):
    minval, minIndex, sOfIJ = find_seam(reduced_image3, emap)
    energy_list3.append(sOfIJ[reduced_image3.shape[0]-1,minIndex])
    disp = remove_seam(disp, minIndex, sOfIJ)
    reduced_image3 = remove_seam(reduced_image3, minIndex, sOfIJ)
    cv2.imwrite(myvariables.orgimg_path, reduced_image3)
    cv2.imwrite(myvariables.dispimg_path, disp)

    data = cv2.imread(myvariables.orgimg_path)
    data = np.array(data, np.float32)
    data -= mean_bgr
    data = data.transpose((2, 0, 1))
    data = torch.from_numpy(data).float().unsqueeze(0)
    if myvariables.cuda:
      data = data.cuda()
    data = Variable(data)
    out = gmodel(data)
    out = [torch.sigmoid(out[-1]).cpu().data.numpy()[0, 0, :, :]]
    gradient_map = out[-1]

    inf_data = InfDataloader(img_path=myvariables.orgimg_path, target_size=256)
    inf_dataloader = DataLoader(inf_data, batch_size=1, shuffle=True, num_workers=2)
    with torch.no_grad():
      for batch_idx, (img_np, img_tor) in enumerate(inf_dataloader, start=1):
        img_tor = img_tor.to(myvariables.device)
        pred_masks, _ = smodel(img_tor)
        img_np = np.squeeze(img_np.numpy(), axis=0)
        img_np = img_np.astype(np.uint8)
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        mask = np.squeeze(pred_masks.cpu().numpy(), axis=(0, 1))
        saliency_map = ReturnToOriginSize(myvariables.orgimg_path, mask)

    depth_map = cv2.imread(myvariables.orgimg_path)
    depth_map = cv2.cvtColor(depth_map, cv2.COLOR_RGB2GRAY)
    depth_map = depth_map/255.0

    g_importance = gradient_map.sum()/(gradient_map.shape[0]*gradient_map.shape[1])
    s_importance = saliency_map.sum()/(saliency_map.shape[0]*saliency_map.shape[1])
    d_importance = depth_map.sum()/(depth_map.shape[0]*depth_map.shape[1])
    g_coe = 4*g_importance
    s_coe = 2.5*s_importance
    d_coe = 1*d_importance
    coe_emap = (g_coe*gradient_map + s_coe*saliency_map + d_coe*depth_map)/(g_coe + s_coe + d_coe)

    emap = coe_emap

  return energy_list3

def remove_vertical_seams(number):
  os.system('cp images/view1.png img_temp/img_temp.png')
  os.system('cp depth_map/disp1.png disp_temp/disp_temp.png')

  gmodel = bdcn.BDCN()
  gmodel.load_state_dict(torch.load(myvariables.gmodel_path, map_location=myvariables.device))
  mean_bgr = np.array([104.00699, 116.66877, 122.67892])
  if myvariables.cuda:
    gmodel.cuda()
  gmodel.eval()
  data = cv2.imread(myvariables._orgimg_path)
  data = np.array(data, np.float32)
  data -= mean_bgr
  data = data.transpose((2, 0, 1))
  data = torch.from_numpy(data).float().unsqueeze(0)
  if myvariables.cuda:
    data = data.cuda()
  data = Variable(data)
  out = gmodel(data)
  out = [torch.sigmoid(out[-1]).cpu().data.numpy()[0, 0, :, :]]
  gradient_map = out[-1]

  smodel = SODModel()
  chkpt = torch.load(myvariables.smodel_path, map_location=myvariables.device)
  smodel.load_state_dict(chkpt['model'])
  smodel.to(myvariables.device)
  smodel.eval()
  inf_data = InfDataloader(img_path=myvariables._orgimg_path, target_size=256)
  inf_dataloader = DataLoader(inf_data, batch_size=1, shuffle=True, num_workers=2)
  with torch.no_grad():
    for batch_idx, (img_np, img_tor) in enumerate(inf_dataloader, start=1):
      img_tor = img_tor.to(myvariables.device)
      pred_masks, _ = smodel(img_tor)
      img_np = np.squeeze(img_np.numpy(), axis=0)
      img_np = img_np.astype(np.uint8)
      img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
      mask = np.squeeze(pred_masks.cpu().numpy(), axis=(0, 1))
      saliency_map = ReturnToOriginSize(myvariables._orgimg_path, mask)

  depth_map = cv2.imread(myvariables._orgimg_path)
  depth_map = cv2.cvtColor(depth_map, cv2.COLOR_RGB2GRAY)
  depth_map = depth_map/255.0

  g_importance = gradient_map.sum()/(gradient_map.shape[0]*gradient_map.shape[1])
  s_importance = saliency_map.sum()/(saliency_map.shape[0]*saliency_map.shape[1])
  d_importance = depth_map.sum()/(depth_map.shape[0]*depth_map.shape[1])
  g_coe = 4*g_importance
  s_coe = 2.5*s_importance
  d_coe = 1*d_importance
  coe_emap = (g_coe*gradient_map + s_coe*saliency_map + d_coe*depth_map)/(g_coe + s_coe + d_coe)

  emap = coe_emap
  reduced_image2 = cv2.imread(myvariables._orgimg_path)
  disp = cv2.imread(myvariables._dispimg_path)
  for i in range(number):
    minval, minIndex, sOfIJ = find_seam(reduced_image2, emap)
    disp = remove_seam(disp, minIndex, sOfIJ)
    reduced_image2 = remove_seam(reduced_image2, minIndex, sOfIJ)
    cv2.imwrite(myvariables.orgimg_path, reduced_image2)
    cv2.imwrite(myvariables.dispimg_path, disp)

    data = cv2.imread(myvariables.orgimg_path)
    data = np.array(data, np.float32)
    data -= mean_bgr
    data = data.transpose((2, 0, 1))
    data = torch.from_numpy(data).float().unsqueeze(0)
    if myvariables.cuda:
      data = data.cuda()
    data = Variable(data)
    out = gmodel(data)
    out = [torch.sigmoid(out[-1]).cpu().data.numpy()[0, 0, :, :]]
    gradient_map = out[-1]

    inf_data = InfDataloader(img_path=myvariables.orgimg_path, target_size=256)
    inf_dataloader = DataLoader(inf_data, batch_size=1, shuffle=True, num_workers=2)
    with torch.no_grad():
      for batch_idx, (img_np, img_tor) in enumerate(inf_dataloader, start=1):
        img_tor = img_tor.to(myvariables.device)
        pred_masks, _ = smodel(img_tor)
        img_np = np.squeeze(img_np.numpy(), axis=0)
        img_np = img_np.astype(np.uint8)
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        mask = np.squeeze(pred_masks.cpu().numpy(), axis=(0, 1))
        saliency_map = ReturnToOriginSize(myvariables.orgimg_path, mask)

    depth_map = cv2.imread(myvariables.orgimg_path)
    depth_map = cv2.cvtColor(depth_map, cv2.COLOR_RGB2GRAY)
    depth_map = depth_map/255.0

    g_importance = gradient_map.sum()/(gradient_map.shape[0]*gradient_map.shape[1])
    s_importance = saliency_map.sum()/(saliency_map.shape[0]*saliency_map.shape[1])
    d_importance = depth_map.sum()/(depth_map.shape[0]*depth_map.shape[1])
    g_coe = 4*g_importance
    s_coe = 2.5*s_importance
    d_coe = 1*d_importance
    coe_emap = (g_coe*gradient_map + s_coe*saliency_map + d_coe*depth_map)/(g_coe + s_coe + d_coe)

    emap = coe_emap

  return reduced_image2

def calc_img_energy(image):
  image = image.astype('float32')
  energy = np.absolute(cv2.Sobel(image, -1, 1, 0)) + np.absolute(cv2.Sobel(image, -1, 0, 1))
  energy_map = np.sum(energy, axis=2)
  return energy_map

def calc_seam_cost_forward(energy_map):
  shape = m, n = energy_map.shape
  e_map = np.copy(energy_map).astype('float32')
  backtrack = np.zeros(shape, dtype=int)
  for i in range(1, m):
    for j in range(0, n):
      if j == 0:
        min_idx = np.argmin(e_map[i - 1, j:j + 2])
        min_cost = e_map[i - 1, j + min_idx]
        e_map[i, j] += min_cost
        backtrack[i, j] = j + min_idx
      else:
        min_idx = np.argmin(e_map[i - 1, j - 1:j + 2])
        min_cost = e_map[i - 1, j + min_idx - 1]
        e_map[i, j] += min_cost
        backtrack[i, j] = j + min_idx - 1
  return (e_map, backtrack)

def find_min_seam(energy_map_forward, backtrack):
  shape = m, n = energy_map_forward.shape
  seam = np.zeros(m, dtype=int)
  idx = np.argmin(energy_map_forward[-1])
  cost = energy_map_forward[-1][idx]
  seam[-1] = idx
  for i in range(m - 2, -1, -1):
    idx = backtrack[i + 1, idx]
    seam[i] = idx
  return seam, cost

def remove_seam2(image, seam):
  m, n, _ = image.shape
  out_image = np.zeros((m, n - 1, 3)).astype(dtype=int)
  for i in range(m):
    j = seam[i]
    out_image[i, :, 0] = np.delete(image[i, :, 0], j)
    out_image[i, :, 1] = np.delete(image[i, :, 1], j)
    out_image[i, :, 2] = np.delete(image[i, :, 2], j)
  return out_image

def insert_seam(image, seam):
  m, n, num_channels = image.shape
  out_image = np.zeros((m, n + 1, 3)).astype(dtype=int)
  for i in range(m):
    j = seam[i]
    for ch in range(num_channels):
      if j == 0:
        out_image[i, j, ch] = image[i, j, ch]
        out_image[i, j + 1:, ch] = image[i, j:, ch]
        out_image[i, j + 1, ch] = (int(image[i, j, ch]) + int(image[i, j + 1, ch])) / int(2)
      elif j + 1 == n:
        out_image[i, :j + 1, ch] = image[i, :j + 1, ch]
        out_image[i, j + 1, ch] = int(image[i, j, ch])
      else:
        out_image[i, :j, ch] = image[i, :j, ch]
        out_image[i, j + 1:, ch] = image[i, j:, ch]
        out_image[i, j, ch] = (int(image[i, j - 1, ch]) + int(image[i, j + 1, ch])) / int(2)
  return out_image

def remove_vertical_seam(image):
  img = np.copy(image)
  energy_map = calc_img_energy(img)
  energy_map_forward, backtrack = calc_seam_cost_forward(energy_map)
  (min_seam, cost) = find_min_seam(energy_map_forward, backtrack)
  img = remove_seam2(img, min_seam)
  return img, cost

def remove_horizontal_seam(image):
  img = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
  energy_map = calc_img_energy(img)
  energy_map_forward, backtrack = calc_seam_cost_forward(energy_map)
  (min_seam, cost) = find_min_seam(energy_map_forward, backtrack)
  img = remove_seam2(img, min_seam)
  img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
  return img, cost

def calc_seam_cost_forward(energy_map):
  shape = m, n = energy_map.shape
  e_map = np.copy(energy_map).astype('float32')
  backtrack = np.zeros(shape, dtype=int)
  for i in range(1, m):
    for j in range(0, n):
      if j == 0:
        min_idx = np.argmin(e_map[i - 1, j:j + 2])
        min_cost = e_map[i - 1, j + min_idx]
        e_map[i, j] += min_cost
        backtrack[i, j] = j + min_idx
      else:
        min_idx = np.argmin(e_map[i - 1, j - 1:j + 2])
        min_cost = e_map[i - 1, j + min_idx - 1]
        e_map[i, j] += min_cost
        backtrack[i, j] = j + min_idx - 1
  return (e_map, backtrack)

def extend_image(number):
  os.system('cp images/view1.png img_temp/img_temp.png')
  os.system('cp depth_map/disp1.png disp_temp/disp_temp.png') 
  image = cv2.imread(myvariables._orgimg_path)
  a = np.arange(0, image.shape[1], 1)
  b = np.expand_dims(a, axis=0)
  pixels_kept = np.repeat(b, image.shape[0], axis=0)
  pixels_removed = np.zeros((image.shape[0],number), dtype=int)
  img = np.copy(image)
  for c in range(number):
    # Find seam to remove
    energy_map = calc_img_energy(img)
    energy_map_forward, backtrack = calc_seam_cost_forward(energy_map)
    (min_seam, cost) = find_min_seam(energy_map_forward, backtrack)
    # Remove minimum seam from ndarray that tracks image reductions and add to list of pixels removed
    rows, cols, _ = img.shape
    mask = np.ones((rows, cols), dtype=np.bool)
    for i in range(0, rows):
        j = min_seam[i]
        mask[i, j] = False
    # Remove seam from image
    pixels_removed[:, c] = pixels_kept[mask == False].reshape((rows,))
    pixels_kept = pixels_kept[mask].reshape((rows, cols - 1))
    img = remove_seam2(img, min_seam)
  pixels_removed.sort(axis=1)
  img = np.copy(image)
  for c in range(number):
    img = insert_seam(img, pixels_removed[:, c])
    pixels_removed[:, c + 1:] = pixels_removed[:, c + 1:] + 1
  return img, energy_map

def seam_carving(h_reduction, w_reduction):
  image = cv2.imread(myvariables._orgimg_path)
  input_h = image.shape[0]
  input_w = image.shape[1]
  output_h = input_h - round(h_reduction*input_h/100)
  output_w = input_w - round(w_reduction*input_w/100)
  flag = 0

  new_w = round((input_w*output_h)/input_h)
  new_h = round((input_h*output_w)/input_w)
  if abs(input_h - new_h) > abs(input_w - new_w):
    scaling_h = output_h 
    scaling_w = new_w
  else:
    scaling_h = new_h
    scaling_w = output_w
  delta_h = abs(output_h - scaling_h)
  delta_w = abs(output_w - scaling_w)
  if scaling_h > output_h:
    img = cv2.imread(myvariables._orgimg_path)
    img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    cv2.imwrite(myvariables._orgimg_path, img)

    disp = cv2.imread(myvariables._dispimg_path)
    disp = cv2.rotate(disp, cv2.ROTATE_90_CLOCKWISE)
    cv2.imwrite(myvariables._dispimg_path, disp)

    flag = 1
    delta = delta_h
  elif scaling_h < output_h:
    img = cv2.imread(myvariables._orgimg_path)
    img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    cv2.imwrite(myvariables._orgimg_path, img)

    disp = cv2.imread(myvariables._dispimg_path)
    disp = cv2.rotate(disp, cv2.ROTATE_90_CLOCKWISE)
    cv2.imwrite(myvariables._dispimg_path, disp)

    flag = 1
    out = extend_image(delta_h)
    return out
  if scaling_w > output_w:
    delta = delta_w
  elif scaling_w < output_w:
    out = extend_image(delta_w)
    return out
 
  elist = find_energy_range(delta)
  elist.sort()
  indexL = round(1*(len(elist)-1)/3)
  indexH = round(2*(len(elist)-1)/3)
  E = (elist[indexH] - elist[indexL])/(indexH - indexL)
  for i in range(indexL, indexH):
    if elist[i + 1] - elist[i] > E:
      break
  
  out = remove_vertical_seams(i)

  if flag == 1:
    out = cv2.rotate(out, cv2.ROTATE_90_COUNTERCLOCKWISE)

    img = cv2.imread(myvariables._orgimg_path)
    img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    cv2.imwrite(myvariables._orgimg_path, img)

    disp = cv2.imread(myvariables._dispimg_path)
    disp = cv2.rotate(disp, cv2.ROTATE_90_COUNTERCLOCKWISE)
    cv2.imwrite(myvariables._dispimg_path, disp)

  out = cv2.resize(out, (output_w, output_h))  

  return out





