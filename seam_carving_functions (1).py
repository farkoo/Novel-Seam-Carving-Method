import scipy
import numpy as np
from skimage import img_as_float
import cv2
from pylab import *
import myvariables
from myfunctions import GM_creator, SM_creator, DM_creator
import os

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

def insert_seam(img, minIndex, sOfIJ):
  rows = img.shape[0]
  columns = img.shape[1]
  if len(img.shape) == 2:
    img = cv2.merge((img, img, img))
  extended_matrix = np.zeros(shape=(rows, columns + 1, 3))
  k = minIndex
  for i in range(rows - 1, -1, -1):
    if i == 0:
      b,g,r = img[i,k,:]/2 + img[i,k+1,:]/2
    elif i == rows - 1:
      b,g,r = img[i,k-1,:]/2 + img[i,k,:]/2
    else:
      b,g,r = img[i,k-1,:]/3 + img[i,k,:]/3 + img[i,k+1,:]/3
    tmp = img[i, :, :] 
    extended_matrix[i, :, 0] = np.insert(tmp, k + 1, r, axis=0)[:,0]
    extended_matrix[i, :, 1] = np.insert(tmp, k + 1, g, axis=0)[:,1]
    extended_matrix[i, :, 2] = np.insert(tmp, k + 1, b, axis=0)[:,2]
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
  return extended_matrix

def produce_emap(flag):
  if flag:
    gradient_map = GM_creator(myvariables._img_folder, myvariables._orgimg_path, myvariables.gmodel_path, myvariables.cuda)
    saliency_map = SM_creator(myvariables._img_folder, myvariables._orgimg_path, myvariables.smodel_path, myvariables.img_size, myvariables.bs, myvariables.device)
    depth_map = DM_creator(myvariables._dispimg_path)
  else:
    gradient_map = GM_creator(myvariables.img_folder, myvariables.orgimg_path, myvariables.gmodel_path, myvariables.cuda)
    saliency_map = SM_creator(myvariables.img_folder, myvariables.orgimg_path, myvariables.smodel_path, myvariables.img_size, myvariables.bs, myvariables.device)
    depth_map = DM_creator(myvariables.dispimg_path)
  g_importance = gradient_map.sum()/(gradient_map.shape[0]*gradient_map.shape[1])
  s_importance = saliency_map.sum()/(saliency_map.shape[0]*saliency_map.shape[1])
  d_importance = depth_map.sum()/(depth_map.shape[0]*depth_map.shape[1])
  g_coe = 2*g_importance
  s_coe = 4*s_importance
  d_coe = 3*d_importance
  coe_emap = (g_coe*gradient_map + s_coe*saliency_map + d_coe*depth_map)/(g_coe + s_coe + d_coe)
  return coe_emap

def find_energy_range1(number):
  energy = produce_emap(1)
  img = cv2.imread(myvariables._orgimg_path)
  energy_list = []
  for i in range(number):
    minval, minIndex, sOfIJ = find_seam(img, energy)
    energy_list.append(sOfIJ[img.shape[0]-1,minIndex])
    img = remove_seam(img, minIndex, sOfIJ)
    energy = remove_seam(energy, minIndex, sOfIJ)
    energy = energy[:,:,0]
  return energy_list

def find_energy_range2(number):
  os.system('cp images/view1.png img_temp/img_temp.png')
  os.system('cp depth_map/disp1.png disp_temp/disp_temp.png')
  emap = produce_emap(0)
  reduced_image3 = cv2.imread(myvariables._orgimg_path)
  disp = cv2.imread(myvariables._dispimg_path)
  energy_list2 = []
  for i in range(number):
    minval, minIndex, sOfIJ = find_seam(reduced_image3, emap)
    energy_list2.append(sOfIJ[reduced_image3.shape[0]-1,minIndex])
    disp = remove_seam(disp, minIndex, sOfIJ)
    reduced_image3 = remove_seam(reduced_image3, minIndex, sOfIJ)
    cv2.imwrite(myvariables.orgimg_path, reduced_image3)
    cv2.imwrite(myvariables.dispimg_path, disp)
    emap = produce_emap(0)
  return energy_list2

def remove_vertical_seams1(number):
  emap = produce_emap(1)
  reduced_image1 = cv2.imread(myvariables._orgimg_path)
  for i in range(number):
    minval, minIndex, sOfIJ = find_seam(reduced_image1, emap)
    reduced_image1 = remove_seam(reduced_image1, minIndex, sOfIJ)
    emap = remove_seam(emap, minIndex, sOfIJ)
    emap = emap[:,:,0]
  return reduced_image1  

def insert_vertical_seams1(number):
  emap = produce_emap(1)
  reduced_image1 = cv2.imread(myvariables._orgimg_path)
  for i in range(number):
    minval, minIndex, sOfIJ = find_seam(reduced_image1, emap)
    reduced_image1 = insert_seam(reduced_image1, minIndex, sOfIJ)
    emap = insert_seam(emap, minIndex, sOfIJ)
    emap = emap[:,:,0]
  return reduced_image1  

def remove_vertical_seams2(number):
  os.system('cp images/view1.png img_temp/img_temp.png')
  os.system('cp depth_map/disp1.png disp_temp/disp_temp.png')
  emap = produce_emap(1)
  reduced_image2 = cv2.imread(myvariables._orgimg_path)
  disp = cv2.imread(myvariables._dispimg_path)
  for i in range(number):
    minval, minIndex, sOfIJ = find_seam(reduced_image2, emap)
    disp = remove_seam(disp, minIndex, sOfIJ)
    reduced_image2 = remove_seam(reduced_image2, minIndex, sOfIJ)
    cv2.imwrite(myvariables.orgimg_path, reduced_image2)
    cv2.imwrite(myvariables.dispimg_path, disp)
    emap = produce_emap(0)
  return reduced_image2

def insert_vertical_seams2(number):
  os.system('cp images/view1.png img_temp/img_temp.png')
  os.system('cp depth_map/disp1.png disp_temp/disp_temp.png')
  emap = produce_emap(1)
  reduced_image2 = cv2.imread(myvariables._orgimg_path)
  disp = cv2.imread(myvariables._dispimg_path)
  for i in range(number):
    minval, minIndex, sOfIJ = find_seam(reduced_image2, emap)
    disp = insert_seam(disp, minIndex, sOfIJ)
    reduced_image2 = insert_seam(reduced_image2, minIndex, sOfIJ)
    cv2.imwrite(myvariables.orgimg_path, reduced_image2)
    cv2.imwrite(myvariables.dispimg_path, disp)
    emap = produce_emap(0)
  return reduced_image2






