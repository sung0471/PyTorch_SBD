#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import imageio
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import torchvision.models as models
from PIL import Image
from scipy.spatial.distance import cosine
from sklearn.decomposition import PCA
import os
import cv2
import pickle
import sys
import scipy

video_file = 'Bayshore Aerial Video Transition Demo.mp4' #'Aircraft Carrier Takeoffs & Landings.mp4'
video_name = str(os.path.splitext(os.path.basename(video_file))[0])

# input video (imageio - sometimes error)
# imageio.plugins.ffmpeg.download()
#video = imageio.get_reader('test.mp4')

print('[INFO] loading video...')
# input video (cv2)
cap = cv2.VideoCapture(video_file)
total_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

video = []
while(cap.isOpened()):
    _, frame_image = cap.read()
    frame_num = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    video.append(frame_image)
    if frame_num == total_frame:
        break

# extract CNN feature for each frame in video
class pretrainedAlexNet(nn.Module):
    def __init__(self):
        super(pretrainedAlexNet, self).__init__()
        self.my_features = models.alexnet(pretrained = True).features

    def forward(self, input):
        feature_map = self.my_features(input)
        return feature_map

print('[INFO] extracting features from frames...')
net = pretrainedAlexNet()

num_frame = 0
for i, im in enumerate(video):
    num_frame += 1

frame_feature_arr = np.zeros((num_frame,256,6,6))

if os.path.isfile(video_name + '.features'):
    with open(video_name + '.features', 'rb') as f:
        frame_feature_arr = pickle.load(f)
else:
    for i, im in enumerate(video):
        print(str(i) + '/' + str(len(video)))
        
        # im_np = np.array(im) # cv2 always return numpy array
        im = Image.fromarray(im).resize((224,224))
        frame = np.array(im)
        
        frame_tensor = torch.from_numpy(frame).transpose(0,1).transpose(0,2).unsqueeze(0).float()
        frame_feature_tensor = net(frame_tensor)
        frame_feature_np = frame_feature_tensor.squeeze(0).data.numpy()
        frame_feature_arr[i] = frame_feature_np

    # compare cosine similarity between all consecutive frames
    frame_feature_arr = frame_feature_arr.reshape((frame_feature_arr.shape[0],-1))

    with open(video_name + '.features', 'wb') as f:
        pickle.dump(frame_feature_arr, f)

print('[INFO] doing PCA and calculate cos similarity...')
def scale(X, x_min, x_max):
    nom = (X-X.min(axis=0))*(x_max-x_min)
    denom = X.max(axis=0) - X.min(axis=0)
    denom[denom==0] = 1
    return x_min + nom/denom

do_PCA = True
visualize_features = False
cos_sim = np.zeros((frame_feature_arr.shape[0]-1))
if visualize_features:
    ''' with PCA '''
    pca = PCA(n_components=100)
    frame_feature_arr_pca = pca.fit(frame_feature_arr).transform(frame_feature_arr)
    features_normalized_matrix = scale(frame_feature_arr_pca.transpose(), 0, 1)
    ''' without PCA '''
    # features_normalized_matrix = scale(frame_feature_arr, 255, 0) #0, 1
    # plt.imshow(features_normalized_matrix, aspect=1, interpolation='none') #aspect='auto' / 'equal'
    # plt.show()

    scipy.misc.imsave('feature.png', features_normalized_matrix)
    print('exit program because you select feature visualization')
    sys.exit(1)
else:
    if do_PCA:
        pca = PCA(n_components=100)
        frame_feature_arr_pca = pca.fit(frame_feature_arr).transform(frame_feature_arr)
        for i in range(frame_feature_arr_pca.shape[0]-1):
            cos_sim[i] = cosine(frame_feature_arr_pca[i+1], frame_feature_arr_pca[i])
    else:
        for i in range(frame_feature_arr.shape[0]-1):
            cos_sim[i] = cosine(frame_feature_arr[i+1], frame_feature_arr[i])

x_index = np.arange(1,frame_feature_arr.shape[0],1)
plt.figure()
plt.title(video_name)
plt.plot(x_index, cos_sim, 'b*')
plt.xlabel('frame in time series')
plt.ylabel('cosine similarity difference')
plt.title('cosine similarity')

boundary_index = np.array([])
for i in range(cos_sim.shape[0]):
    if cos_sim[i] > 0.2:
        boundary_index = np.append(boundary_index, i)

# make videos
print('[INFO] creating shot videos...')
boundary_index = np.concatenate((np.array([0]),boundary_index))
boundary_index = np.concatenate((boundary_index, np.array([num_frame])))
# fps = video.get_meta_data()['fps'] # only for imageio

with open(video_name + '.srt', 'w', encoding='utf-8') as f:
    for bound_ind in range(boundary_index.shape[0]-1):
        starttime = float(boundary_index[bound_ind] / fps)
        endtime = float(boundary_index[bound_ind + 1] / fps)
        f.write(str(bound_ind) + '\n')
        f.write(str(starttime) + ' --> ' + str(endtime) + '\n')
        f.write('shot# ' + str(bound_ind) + ' & frame# ' + str(boundary_index[bound_ind + 1]) + '\n')
        f.write('\n')

# if not os.path.exists(video_name + '_output_shots/'):
#     os.makedirs(video_name + '_output_shots/')

# writer_list = []
# for bound_ind in range(boundary_index.shape[0]-1):
#     print(bound_ind)

#     writer = imageio.get_writer(video_name + '_output_shots/'+str(bound_ind)+'.mp4', fps=fps)
#     for i, im in enumerate(video):
#         if i > boundary_index[bound_ind]:
#             writer.append_data(im)
#         if i == boundary_index[bound_ind+1]-1:
#             break
#     writer.close()

plt.show()

print('[INFO] done')