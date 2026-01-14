# import os
# import time
#
# import cv2
# import h5py
# import numpy as np
# import scipy.io
# import scipy.spatial
# # from scipy.ndimage.filters import gaussian_filter
# from scipy.ndimage import gaussian_filter
# import math
# import torch
#
# import glob
# '''change your dataset'''
# # root = '/home/qy/snap/Image-level-Counting/datasets/UCF-QNRF_ECCV18'
#
# root = '/amax/zs/code/WSCC_TAF-main/datasets/UCF-QNRF_ECCV18'
#
# img_train_path = root + '/Train/'
# img_test_path = root + '/Test/'
#
#
# save_train_img_path = root + '/train_data/images/'
# save_test_img_path = root + '/test_data/images/'
#
# os.makedirs(save_train_img_path,exist_ok=True)
# os.makedirs(save_test_img_path,exist_ok=True)
# os.makedirs(save_train_img_path.replace('images','gt_density_map'),exist_ok=True)
# os.makedirs(save_test_img_path.replace('images','gt_density_map'),exist_ok=True)
#
# distance = 1
# img_train = []
# img_test = []
#
#
# path_sets = [img_train_path, img_test_path]
#
# img_paths = []
# for path in path_sets:
#     for img_path in glob.glob(os.path.join(path, '*.jpg')):
#         img_paths.append(img_path)
#
# img_paths.sort()
#
#
# for img_path in img_paths:
#
#
#     Img_data = cv2.imread(img_path)
#     Gt_data = scipy.io.loadmat(img_path.replace('.jpg', '_ann.mat'))
#
#     Gt_data = Gt_data['annPoints']
#
#     if Img_data.shape[1] >= Img_data.shape[0]:
#         rate_1 = 1152.0 / Img_data.shape[1]
#         rate_2 = 768 / Img_data.shape[0]
#         Img_data = cv2.resize(Img_data, (0, 0), fx=rate_1, fy=rate_2)
#         Gt_data[:, 0] = Gt_data[:, 0] * rate_1
#         Gt_data[:, 1] = Gt_data[:, 1] * rate_2
#
#     elif Img_data.shape[0] > Img_data.shape[1]:
#         rate_1 = 1152.0 / Img_data.shape[0]
#         rate_2 = 768.0 / Img_data.shape[1]
#         Img_data = cv2.resize(Img_data, (0, 0), fx=rate_2, fy=rate_1)
#         Gt_data[:, 0] = Gt_data[:, 0] * rate_2
#         Gt_data[:, 1] = Gt_data[:, 1] * rate_1
#         print(img_path)
#
#
#     kpoint = np.zeros((Img_data.shape[0], Img_data.shape[1]))
#     for count in range(0, len(Gt_data)):
#         if int(Gt_data[count][1]) < Img_data.shape[0] and int(Gt_data[count][0]) < Img_data.shape[1]:
#             kpoint[int(Gt_data[count][1]), int(Gt_data[count][0])] = 1
#
#     height, width = Img_data.shape[0], Img_data.shape[1]
#
#     m = int(width / 384)
#     n = int(height / 384)
#     fname = img_path.split('/')[-1]
#     root_path = img_path.split('img_')[0]
#
#
#     if root_path.split('/')[-2] == 'Train':
#
#         for i in range(0, m):
#             for j in range(0, n):
#                 crop_img = Img_data[j * 384: 384 * (j + 1), i * 384:(i + 1) * 384, ]
#                 crop_kpoint = kpoint[j * 384: 384 * (j + 1), i * 384:(i + 1) * 384, ]
#                 gt_count = np.sum(crop_kpoint)
#
#                 save_fname = str(i) + str(j) + str('_') + fname
#                 save_path = root_path.replace('Train', 'train_data/images') + save_fname
#                 h5_path = save_path.replace('.jpg', '.h5').replace('images', 'gt_density_map')
#
#                 with h5py.File(h5_path, 'w') as hf:
#                     hf['gt_count'] = gt_count
#                 cv2.imwrite(save_path, crop_img)
#
#                 density_map = gaussian_filter(crop_kpoint, 2)
#                 density_map = density_map
#                 density_map = density_map / np.max(density_map) * 255
#                 density_map = density_map.astype(np.uint8)
#                 density_map = cv2.applyColorMap(density_map, 2)
#                 result = np.hstack((density_map, crop_img))
#                 cv2.imwrite(save_path.replace('images', 'gt_show').replace('jpg', 'jpg'), result)
#
#     else:
#         save_path = root_path.replace('Test', 'test_data/images') + fname
#         print(save_path)
#         cv2.imwrite(save_path, Img_data)
#
#         gt_count = np.sum(kpoint)
#         with h5py.File(save_path.replace('.jpg', '.h5').replace('images', 'gt_density_map'), 'w') as hf:
#             hf['gt_count'] = gt_count


import os
import time

import cv2
import h5py
import numpy as np
import scipy.io
import scipy.spatial
# from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage import gaussian_filter
import math
import torch

import glob

'''change your dataset'''
# root = '/home/qy/snap/Image-level-Counting/datasets/UCF-QNRF_ECCV18'

root = '/amax/zs/code/WSCC_TAF-main/datasets/UCF-QNRF_ECCV18'

img_train_path = root + '/Train/'
img_test_path = root + '/Test/'

save_train_img_path = root + '/train_data/images/'
save_test_img_path = root + '/test_data/images/'

os.makedirs(save_train_img_path, exist_ok=True)
os.makedirs(save_test_img_path, exist_ok=True)
os.makedirs(save_train_img_path.replace('images', 'gt_density_map'), exist_ok=True)
os.makedirs(save_test_img_path.replace('images', 'gt_density_map'), exist_ok=True)
os.makedirs(save_train_img_path.replace('images', 'gt_show'), exist_ok=True)  # 添加这一行，确保gt_show目录存在

distance = 1
img_train = []
img_test = []

path_sets = [img_train_path, img_test_path]

img_paths = []
for path in path_sets:
    for img_path in glob.glob(os.path.join(path, '*.jpg')):
        img_paths.append(img_path)

img_paths.sort()

for img_path in img_paths:

    Img_data = cv2.imread(img_path)
    Gt_data = scipy.io.loadmat(img_path.replace('.jpg', '_ann.mat'))

    Gt_data = Gt_data['annPoints']

    if Img_data.shape[1] >= Img_data.shape[0]:
        rate_1 = 1152.0 / Img_data.shape[1]
        rate_2 = 768 / Img_data.shape[0]
        Img_data = cv2.resize(Img_data, (0, 0), fx=rate_1, fy=rate_2)
        Gt_data[:, 0] = Gt_data[:, 0] * rate_1
        Gt_data[:, 1] = Gt_data[:, 1] * rate_2

    elif Img_data.shape[0] > Img_data.shape[1]:
        rate_1 = 1152.0 / Img_data.shape[0]
        rate_2 = 768.0 / Img_data.shape[1]
        Img_data = cv2.resize(Img_data, (0, 0), fx=rate_2, fy=rate_1)
        Gt_data[:, 0] = Gt_data[:, 0] * rate_2
        Gt_data[:, 1] = Gt_data[:, 1] * rate_1
        print(img_path)

    kpoint = np.zeros((Img_data.shape[0], Img_data.shape[1]))
    for count in range(0, len(Gt_data)):
        if int(Gt_data[count][1]) < Img_data.shape[0] and int(Gt_data[count][0]) < Img_data.shape[1]:
            kpoint[int(Gt_data[count][1]), int(Gt_data[count][0])] = 1

    height, width = Img_data.shape[0], Img_data.shape[1]

    m = int(width / 384)
    n = int(height / 384)
    fname = img_path.split('/')[-1]
    root_path = img_path.split('img_')[0]

    if root_path.split('/')[-2] == 'Train':

        for i in range(0, m):
            for j in range(0, n):
                crop_img = Img_data[j * 384: 384 * (j + 1), i * 384:(i + 1) * 384, ]
                crop_kpoint = kpoint[j * 384: 384 * (j + 1), i * 384:(i + 1) * 384, ]
                gt_count = np.sum(crop_kpoint)

                save_fname = str(i) + str(j) + str('_') + fname
                save_path = root_path.replace('Train', 'train_data/images') + save_fname
                h5_path = save_path.replace('.jpg', '.h5').replace('images', 'gt_density_map')

                with h5py.File(h5_path, 'w') as hf:
                    hf['gt_count'] = gt_count
                cv2.imwrite(save_path, crop_img)

                density_map = gaussian_filter(crop_kpoint, 2)

                # 修复：检查密度图最大值是否为0，避免除以零
                max_val = np.max(density_map)
                if max_val > 0:
                    density_map = density_map / max_val * 255
                else:
                    # 如果最大值为0，直接返回全零的密度图
                    density_map = np.zeros_like(density_map)

                density_map = density_map.astype(np.uint8)
                density_map = cv2.applyColorMap(density_map, 2)
                result = np.hstack((density_map, crop_img))
                save_gt_show_path = save_path.replace('images', 'gt_show').replace('jpg', 'jpg')
                os.makedirs(os.path.dirname(save_gt_show_path), exist_ok=True)  # 确保目录存在
                cv2.imwrite(save_gt_show_path, result)

    else:
        save_path = root_path.replace('Test', 'test_data/images') + fname
        print(save_path)
        cv2.imwrite(save_path, Img_data)

        gt_count = np.sum(kpoint)
        with h5py.File(save_path.replace('.jpg', '.h5').replace('images', 'gt_density_map'), 'w') as hf:
            hf['gt_count'] = gt_count
