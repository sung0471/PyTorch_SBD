import os
import json
import cv2
from PIL import Image

import xml.etree.ElementTree as elemTree
from shutil import copy2
from utils.time_control import TimeControl

dataset_video_origin = 'E:/video/TRECVID/2007/shot.test'
dataset_video_root = 'videos/test'
dataset_annotation_dir = 'annotations'
if not os.path.exists(dataset_video_root):
    os.makedirs(dataset_video_root)
if not os.path.exists(dataset_video_root):
    os.makedirs(dataset_video_root)
if not os.path.exists(dataset_annotation_dir):
    os.makedirs(dataset_annotation_dir)

def make_dataset_TRECVID():
    xml_tree = elemTree.parse('ref/shotBoundaryReferenceFiles.xml')
    timer_total = TimeControl()
    timer_video = TimeControl()
    gts = dict()
    timer_log = dict()
    
    timer_total.timer_start()
    for elem in xml_tree.findall('./referenceFileName'):
        annotation_file_name = elem.get('segName')
        xml_file_path = os.path.join('ref', annotation_file_name)
        video_name = elem.get('videoName')
    
        gts[video_name] = {'transitions': list(), 'frame_num': float()}
        timer_log[video_name] = {'xml_parsing': '', 'video_copy': ''}
    
        timer_video.timer_start()
        print('Check {}'.format(xml_file_path))
        if os.path.exists(xml_file_path):
            annotation_xml_tree = elemTree.parse(xml_file_path)
            top_node = annotation_xml_tree.getroot()
            gts[video_name]['frame_num'] = int(top_node.get('totalFNum'))
            for transition_elem in top_node.findall('trans'):
                transition = [int(transition_elem.get('preFNum')), int(transition_elem.get('postFNum'))]
                gts[video_name]['transitions'].append(transition)
            timer_log[video_name]['xml_parsing'] = timer_video.timer_log()
    
            origin = os.path.join(dataset_video_origin, video_name)
            if not os.path.exists(dataset_video_root + '/' + video_name):
                copy2(origin, dataset_video_root)
            timer_log[video_name]['video_copy'] = timer_video.timer_end()
        else:
            print('There is no file : {}'.format(xml_file_path))
    
    timer_log['total'] = timer_total.timer_end()
    
    gts_path = os.path.join(dataset_annotation_dir, 'test.json')
    timer_path = 'timer.json'
    video_list_path = os.path.join('video_lists', 'test.txt')
    if not os.path.exists('video_lists'):
        os.makedirs('video_lists')
    
    json.dump(gts, open(gts_path, 'w'), indent=1, ensure_ascii=False)
    json.dump(timer_log, open(timer_path, 'w'), indent=1, ensure_ascii=False)
    with open(video_list_path, 'wt', encoding='utf-8') as f:
        for video_name in gts.keys():
            f.write(video_name + '\n')


def check_10_frames(video_name):
    video_path = os.path.join(dataset_video_root, video_name)
    video_cap = cv2.VideoCapture(video_path)
    attr_name = ['CV_CAP_PROP_POS_MSEC', 'CV_CAP_PROP_POS_FRAMES', 'CV_CAP_PROP_POS_AVI_RATIO',
                 'CV_CAP_PROP_FRAME_WIDTH', 'CV_CAP_PROP_FRAME_HEIGHT', 'CV_CAP_PROP_FPS', 'CV_CAP_PROP_FOURCC',
                 'CV_CAP_PROP_FRAME_COUNT', 'CV_CAP_PROP_FORMAT', 'CV_CAP_PROP_MODE', 'CV_CAP_PROP_BRIGHTNESS',
                 'CV_CAP_PROP_CONTRAST', 'CV_CAP_PROP_SATURATION', 'CV_CAP_PROP_HUE', 'CV_CAP_PROP_GAIN',
                 'CV_CAP_PROP_EXPOSURE', 'CV_CAP_PROP_CONVERT_RGB', 'CV_CAP_PROP_WHITE_BALANCE_U',
                 'CV_CAP_PROP_WHITE_BALANCE_V', 'CV_CAP_PROP_RECTIFICATION', 'CV_CAP_PROP_ISO_SPEED',
                 'CV_CAP_PROP_BUFFERSIZE']
    for num, info in enumerate(attr_name):
        print('{}: {}'.format(info, video_cap.get(num)), end='\t')
        if num != 0 and num % 3 == 0:
            print()
    for i in range(10):
        _, frame = video_cap.read()
        print('#{} : {}'.format(i, frame.sum()))


if __name__ == '__main__':
    make_dataset_TRECVID()
    check_10_frames('BG_11362.mpg')
