from scipy.spatial.distance import cosine
from sklearn.decomposition import PCA
import cv2
import os
import pickle
import matplotlib.pyplot as plt
import scipy
import sys

from models.squeezenet import SqueezeNetFeature
from lib.spatial_transforms import *

def candidate_extraction(root_dir, video_file, device):
    video_name = str(os.path.splitext(video_file)[0])
    video_full_name = os.path.join(root_dir,video_name)
    video_dir = os.path.join(root_dir,video_file)

    print('[INFO] loading video...')
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

    frame_feature_arr = np.zeros((num_frame, 512, 7, 7))

    # 19.4.9. add
    model = SqueezeNetFeature().cuda().to(device)
    print(model)

    if os.path.isfile(video_full_name + '.features'):
        print('[INFO] loading video feature ...')
        with open(video_full_name + '.features', 'rb') as f:
            frame_feature_arr = pickle.load(f)
    else:
        for i, im in enumerate(video):
            if i % 500 == 0:
                print(str(i) + '/' + str(len(video)))

            # im_np = np.array(im) # cv2 always return numpy array
            im = Image.fromarray(im).resize((128, 128))
            frame = np.array(im)

            frame_tensor = torch.from_numpy(frame).transpose(0, 1).transpose(0, 2).unsqueeze(0).float()
            frame_feature_tensor = model(frame_tensor).cpu()
            frame_feature_np = frame_feature_tensor.squeeze(0).data.numpy()
            frame_feature_arr[i] = frame_feature_np

        # compare cosine similarity between all consecutive frames
        frame_feature_arr = frame_feature_arr.reshape((frame_feature_arr.shape[0], -1))

        with open(video_full_name + '.features', 'wb') as f:
            print('[INFO] saving video feature pickle...')
            pickle.dump(frame_feature_arr, f)

    print('[INFO] doing PCA and calculate cos similarity...')

    def scale(X, x_min, x_max):
        nom = (X - X.min(axis=0)) * (x_max - x_min)
        denom = X.max(axis=0) - X.min(axis=0)
        denom[denom == 0] = 1
        return x_min + nom / denom

    do_PCA = True
    visualize_features = False
    cos_sim = np.zeros((frame_feature_arr.shape[0] - 1))
    if visualize_features:
        ''' with PCA '''
        pca = PCA(n_components=100)
        frame_feature_arr_pca = pca.fit(frame_feature_arr).transform(frame_feature_arr)
        features_normalized_matrix = scale(frame_feature_arr_pca.transpose(), 0, 1)
        ''' without PCA '''
        # features_normalized_matrix = scale(frame_feature_arr, 255, 0) #0, 1
        # plt.imshow(features_normalized_matrix, aspect=1, interpolation='none') #aspect='auto' / 'equal'
        # plt.show()

        scipy.misc.imsave(video_full_name + '.feature.png', features_normalized_matrix)
        print('exit program because you select feature visualization')
        sys.exit(1)
    else:
        if do_PCA:
            pca = PCA(n_components=100)
            frame_feature_arr_pca = pca.fit(frame_feature_arr).transform(frame_feature_arr)
            for i in range(frame_feature_arr_pca.shape[0] - 1):
                cos_sim[i] = cosine(frame_feature_arr_pca[i + 1], frame_feature_arr_pca[i])
        else:
            for i in range(frame_feature_arr.shape[0] - 1):
                cos_sim[i] = cosine(frame_feature_arr[i + 1], frame_feature_arr[i])

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
    for i in range(cos_sim.shape[0]):
        if cos_sim[i] > 0.2:
            boundary_index = np.append(boundary_index, i)

    # make videos
    print('[INFO] creating shot videos srt...')
    boundary_index = np.concatenate((np.array([0]), boundary_index))
    boundary_index = np.concatenate((boundary_index, np.array([num_frame])))
    # fps = video.get_meta_data()['fps'] # only for imageio

    do_srt = True
    if do_srt:
        with open(video_full_name + '.srt', 'w', encoding='utf-8') as f:
            for bound_ind in range(boundary_index.shape[0] - 1):
                starttime = float(boundary_index[bound_ind] / fps)
                endtime = float(boundary_index[bound_ind + 1] / fps)
                f.write(str(bound_ind) + '\n')
                f.write(str(starttime) + ' --> ' + str(endtime) + '\n')
                f.write('shot# ' + str(bound_ind) + ' & frame# ' + str(boundary_index[bound_ind + 1]) + '\n')
                f.write('\n')

    return video, num_frame, fps, boundary_index


if __name__ == '__main__':
    root_dir = 'misaeng_test'
    video_file = '1001.0001.0001.0001.0008.mp4'
    video_name = str(os.path.splitext(video_file)[0])
    video_full_name = os.path.join(root_dir, video_name)

    # input video (imageio - sometimes error)
    # imageio.plugins.ffmpeg.download()
    # video = imageio.get_reader('test.mp4')

    print('[INFO] extracting candidate')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    video, num_frame, fps, boundary_index = candidate_extraction(root_dir, video_file, device)
