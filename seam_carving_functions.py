import scipy
import numpy as np
from skimage import img_as_float
import cv2
from pylab import *

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


def find_energy_range(img, energy, number):
  energy_list = []
  for i in range(number):
    minval, minIndex, sOfIJ = find_seam(img, energy)
    energy_list.append(sOfIJ[img.shape[0]-1,minIndex])
    img = remove_seam(img, minIndex, sOfIJ)
    energy = remove_seam(energy, minIndex, sOfIJ)
    energy = energy[:,:,0]
  return energy_list