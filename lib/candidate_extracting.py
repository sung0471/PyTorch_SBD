from scipy.spatial.distance import cosine
from sklearn.decomposition import PCA
import cv2
import os
import pickle
import matplotlib.pyplot as plt
import scipy
import sys
import json

from models.squeezenet import SqueezeNetFeature
from lib.spatial_transforms import *


def candidate_extraction(root_dir, video_file, model, adjacent=True):
    video_name = str(os.path.splitext(video_file)[0])
    video_full_name = os.path.join(root_dir, video_name)
    video_dir = os.path.join(root_dir, video_file)

    # print('[INFO] loading video...', flush=True)
    # input video (cv2)
    cap = cv2.VideoCapture(video_dir)
    total_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    video = []
    while (cap.isOpened()):
        _, frame_image = cap.read()
        frame_num = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        video.append(frame_image)
        if frame_num == total_frame:
            break

    num_frame = 0
    for i, im in enumerate(video):
        num_frame += 1

    # 19.4.9. add
    # model = SqueezeNetFeature().cuda(device)
    # print(model)

    if adjacent:
        feature_path = video_full_name + '.features'
    else:
        feature_path = video_full_name + '.no_adjacent.features'

    threshold = 32
    if os.path.isfile(feature_path):
        # print('[INFO] loading video feature ...', flush=True)
        with open(feature_path, 'rb') as f:
            cos_sim = pickle.load(f)
    else:
        frame_feature_arr = np.zeros((num_frame, 512, 7, 7))
        for i, im in enumerate(video):
            # if i % 500 == 0:
            #     print(str(i) + '/' + str(len(video)), flush=True)

            # im_np = np.array(im) # cv2 always return numpy array
            im = Image.fromarray(im).resize((128, 128))
            frame = np.array(im)

            frame_tensor = torch.from_numpy(frame).transpose(0, 1).transpose(0, 2).unsqueeze(0).float()
            frame_feature_tensor = model(frame_tensor).cpu()
            frame_feature_np = frame_feature_tensor.squeeze(0).data.numpy()
            frame_feature_arr[i] = frame_feature_np

        # compare cosine similarity between all consecutive frames
        frame_feature_arr = frame_feature_arr.reshape((frame_feature_arr.shape[0], -1))

        print('[INFO] doing PCA and calculate cos similarity...', flush=True)

        def scale(X, x_min, x_max):
            nom = (X - X.min(axis=0)) * (x_max - x_min)
            denom = X.max(axis=0) - X.min(axis=0)
            denom[denom == 0] = 1
            return x_min + nom / denom

        do_PCA = True
        if do_PCA:
            pca = PCA(n_components=100)
            frame_feature_arr_new = pca.fit(frame_feature_arr).transform(frame_feature_arr)
        else:
            frame_feature_arr_new = frame_feature_arr

        visualize_features = False
        cos_sim = np.zeros((frame_feature_arr.shape[0] - 1))
        if visualize_features:
            ''' with PCA '''
            features_normalized_matrix = scale(frame_feature_arr_new.transpose(), 0, 1)
            ''' without PCA '''
            # features_normalized_matrix = scale(frame_feature_arr, 255, 0) #0, 1
            # plt.imshow(features_normalized_matrix, aspect=1, interpolation='none') #aspect='auto' / 'equal'
            # plt.show()

            scipy.misc.imsave(feature_path + '.feature.png', features_normalized_matrix)
            print('exit program because you select feature visualization', flush=True)
            sys.exit(1)
        else:
            if adjacent:
                for i in range(frame_feature_arr_new.shape[0] - 1):
                    cos_sim[i] = cosine(frame_feature_arr_new[i + 1], frame_feature_arr_new[i])
            else:
                i = 0
                while i+threshold < frame_feature_arr_new.shape[0]:
                    cos_sim[i] = cosine(frame_feature_arr_new[i + threshold], frame_feature_arr_new[i])
                    i += 1

        with open(feature_path, 'wb') as f:
            print('[INFO] saving video feature pickle...', flush=True)
            pickle.dump(cos_sim, f)

    do_figure = False
    if do_figure:
        x_index = np.arange(1, frame_feature_arr.shape[0], 1)
        plt.figure()
        plt.title(video_name)
        plt.plot(x_index, cos_sim, 'b*')
        plt.xlabel('frame in time series')
        plt.ylabel('cosine similarity difference')
        plt.title('cosine similarity')
        plt.show()

    boundary_index = np.array([])
    new_arr = []
    new_boundary_index = []
    if adjacent:
        for i in range(cos_sim.shape[0]):
            if cos_sim[i] > 0.2:
                boundary_index = np.append(boundary_index, i)
        # new_arr.append(list(cos_sim))
        # new_arr.append(list(boundary_index))
        # json.dump(new_arr, open(os.path.join(root_dir, video_name + '.adjacent.json'), 'w'), indent=1)
        # scipy.misc.imsave(feature_path + '.adjacent.feature.png', torch.Tensor(cos_sim).unsqueeze(0))

    else:
        for i in range(cos_sim.shape[0]):
            if cos_sim[i] > 0.2:
                boundary_index = np.append(boundary_index, i + threshold/2 - 1)
        last = -1
        for boundary in boundary_index:
            if boundary - last == 1:
                if len(new_boundary_index[-1]) == 1:
                    new_boundary_index[-1].append(boundary)
                else:
                    new_boundary_index[-1][-1] = boundary
            else:
                new_arr_element = []
                new_boundary_index.append(new_arr_element)
                new_boundary_index[-1].append(boundary)
            last = boundary

        # new_arr.append(list(cos_sim))
        # new_arr.append(list(boundary_index))
        # new_arr.append(list(new_boundary_index))
        # json.dump(new_arr, open(os.path.join(root_dir, video_name + '.no_adjacent.json'), 'w'), indent=1)
        # scipy.misc.imsave(feature_path + '.no_adjacent.feature.png', torch.Tensor(cos_sim).unsqueeze(0))

    # make videos
    boundary_index = np.concatenate((np.array([0]), boundary_index))
    boundary_index = np.concatenate((boundary_index, np.array([num_frame])))
    # fps = video.get_meta_data()['fps'] # only for imageio

    do_srt = False
    if do_srt:
        print('[INFO] creating shot videos srt...', flush=True)
        with open(feature_path + '.srt', 'w', encoding='utf-8') as f:
            if adjacent:
                for bound_ind in range(boundary_index.shape[0] - 1):
                    starttime = float(boundary_index[bound_ind] / fps)
                    endtime = float(boundary_index[bound_ind + 1] / fps)
                    f.write(str(bound_ind) + '\n')
                    f.write(str(starttime) + ' --> ' + str(endtime) + '\n')
                    f.write('shot# ' + str(bound_ind) + ' & frame# ' + str(boundary_index[bound_ind + 1]) + '\n')
                    f.write('\n')
            else:
                for bound_1, bound_2 in new_boundary_index:
                    starttime = float(bound_1 / fps)
                    endtime = float(bound_2 / fps)
                    f.write(str(starttime) + ' --> ' + str(endtime) + '\n')
                    f.write('transitive# ' + str(bound_1) + ' ~ ' + str(bound_2) + '\n')
                    f.write('\n')

    return boundary_index, total_frame, fps


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    root_dir = '../misaeng_test'
    video_file = '1001.0001.0001.0001.0008.mp4'
    video_name = str(os.path.splitext(video_file)[0])
    video_full_name = os.path.join(root_dir, video_name)

    # input video (imageio - sometimes error)
    # imageio.plugins.ffmpeg.download()
    # video = imageio.get_reader('test.mp4')

    print('[INFO] extracting candidate')
    model = SqueezeNetFeature()
    boundary_index_adj, total_length, fps = candidate_extraction(root_dir, video_file, model, adjacent=True)
    boundary_index_no_adj, total_length, fps = candidate_extraction(root_dir, video_file, model, adjacent=False)
    print(boundary_index_adj)
    print(boundary_index_no_adj)

    result = {}
    target = [112, 170, 289, 342, 394, 420, 443, 489, 532, 675, 710, 770, 815, 835, 853, 872, 900, 947, 972, 993, 1008,
              1023, 1050, 1099, 1138, 1183, 1228, 1265, 1299, 1347, 1382, 1407, 1444, 1499, 1529, 1570, 1605]
    result["target"] = [_+1 for _ in target]
    result["adjacent"] = len(boundary_index_adj)
    result["no_adjacent"] = len(boundary_index_no_adj)

    total_length = len(result["target"])
    correct_length_adj = 0
    correct_length_no_adj = 0
    for frame in boundary_index_adj:
        if frame in result["target"]:
            correct_length_adj += 1
    for frame in boundary_index_no_adj:
        if frame in result["target"]:
            correct_length_no_adj += 1

    print("adjacent({}) : {}/{} = {}".format(
        result["adjacent"], correct_length_adj, total_length, correct_length_adj / total_length))
    print("no adjacent({}) : {}/{} = {}".format(
        result["no_adjacent"], correct_length_no_adj, total_length, correct_length_no_adj / total_length))

    # json.dump(result, open(os.path.join(root_dir,video_name+'.json'),'w'), indent=1)

    # frame_index_list = no_candidate_frame_pos(root_dir, video_file, 16)
    # print(frame_index_list)
