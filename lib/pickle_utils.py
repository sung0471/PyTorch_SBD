import pickle
import os


class PickleUtils:
    def __init__(self, opt, video_name_list, data_name_list):
        model = 'KD' if opt.loss_type == 'KDloss' else opt.model
        KD_type = '{}+{}'.format(opt.model, opt.teacher_model) if opt.loss_type == 'KDloss' else None
        is_pretrained = 'pretrained' if opt.pretrained_model else 'no_pretrained'
        epoch = 'epoch_' + str(opt.epoch)
        is_full_data = '.full' if opt.is_full_data else '.no_full'

        root_dir = os.path.join(opt.result_dir, 'test_pickle')
        model_dir = os.path.join(root_dir, model)
        if KD_type is not None:
            model_dir = os.path.join(model_dir, KD_type)
        pretrained_dir = os.path.join(model_dir, is_pretrained)
        pickle_dir = os.path.join(pretrained_dir, epoch)

        if not os.path.exists(pickle_dir):
            os.makedirs(pickle_dir)

        pickle_path_dict = dict()
        for video_name in video_name_list:
            pickle_path_dict[video_name] = list()
            for data_name in data_name_list:
                pickle_path_dict[video_name] += [os.path.join(pickle_dir, video_name + is_full_data + '.' + data_name)]

        self.pickle_dir_list = pickle_path_dict

    def get_pickle_dir(self, video_name):
        return self.pickle_dir_list[video_name]

    def check_pickle_data(self, video_name):
        for i, path in enumerate(self.pickle_dir_list[video_name]):
            if os.path.exists(path):
                return False
        return True

    def save_pickle(self, video_name, data_list):
        for i, path in enumerate(self.pickle_dir_list[video_name]):
            with open(path, 'wb') as f:
                pickle.dump(data_list[i], f)

    def load_pickle(self, video_name):
        data_list = list()
        for i, path in enumerate(self.pickle_dir_list[video_name]):
            with open(path, 'rb') as f:
                data_list += [pickle.load(f)]

        return data_list
