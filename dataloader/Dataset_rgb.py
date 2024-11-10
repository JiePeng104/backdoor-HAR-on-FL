import json
import os
import select
import sys

import math
import cv2
import numpy as np
import torch
import torchvision.transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader


class UCF101(Dataset):
    """
    Args:
        class_idxs (string): Path to list of class names and corresponding label (.txt)
        split (string): Path to train or test split (.txt)
        frames_root (string): Directory (root) of directories (classes) of directories (vid_id) of frames.
        UCF101
        ├── ApplyEyeMakeup
        │   ├── v_ApplyEyeMakeup_g01_c01
        │   │   ├── 000.jpg
        │   │   ├── 001.jpg
        │   │   └── ...
        │   └── ...
        │
        ├── ApplyLipstick
        │   ├── v_ApplyLipstick_g01_c01
        │   │   ├── 000.jpg
        │   │   ├── 001.jpg
        │   │   └── ...
        │   └── ...
        │
        ├── Archery
        │   │   ├── 000.jpg
        │   │   ├── 001.jpg
        │   │   └── ...
        │   └── ...
        │   └── ...
        │
        clip_len (int): Number of frames per sample, i.e. depth of Model input. Default is 16
        train (bool): Training vs. Testing model. Default is True
        flip(bool): If randomly flip video frames. Default is -1 which means stay
        poi_tar(int): target class. if poi_tar = -1 the label of each video will not change, else label would be poi_tar. Default is -1

    """
    def __init__(self, class_idxs, split, frames_root, train=True, clip_len=16, flip=True, poi_tar=-1, adv=False):

        self.class_idxs = class_idxs
        self.split_path = split
        self.frames_root = frames_root
        self.train = train
        self.clip_len = clip_len
        self.flip = flip
        self.poi_tar = poi_tar
        self.class_dict = self.read_class_ind()
        self.paths = self.read_split()
        self.data_list = self.build_data_list()
        self.crop_size = 112
        self.resize_height = 224
        self.resize_width = 224
        self.adv = adv
        self.resize = torchvision.transforms.Resize(size=(self.resize_height, self.resize_width))
        flip = torchvision.transforms.RandomHorizontalFlip(1)
        self.s_transform = torchvision.transforms.Compose([self.resize, flip])

    # Reads .txt file w/ each line formatted as "1 ApplyEyeMakeup" and returns dictionary {'ApplyEyeMakeup': 0, ...}
    def read_class_ind(self):
        class_dict = {}
        with open(self.class_idxs) as f:
            for line in f:
                label, class_name_key = line.strip().split()
                if class_name_key not in class_dict:
                    class_dict[class_name_key] = []
                class_dict[class_name_key] = int(label) - 1  # .append(line.strip())

        return class_dict

    # Reads train or test split.txt file and returns list [('v_ApplyEyeMakeup_g08_c01', array(0)), ...]
    def read_split(self):
        paths = []
        with open(self.split_path) as f:
            for line in f:
                rel_vid_path = line.strip().split('.')[0]
                # ApplyEyeMakeup/v_ApplyEyeMakeup_g08_c01
                class_name, vid_id = rel_vid_path.split('/')
                # ApplyEyeMakeup, v_ApplyEyeMakeup_g08_c01
                vid_dir = os.path.join(self.frames_root, rel_vid_path)
                # /datasets/UCF-101/Frames/frames-128x128/ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c01
                assert os.path.exists(vid_dir), 'Directory %s does not exist!' % vid_dir
                paths.append((vid_dir, class_name))
        return paths

    def build_data_list(self):
        paths = self.paths
        class_dict = self.class_dict
        data_list = []
        for vid_dir, class_name in paths:
            if self.poi_tar != -1:
                label = np.array(self.poi_tar, dtype=int)
            else:
                label = np.array(class_dict[class_name], dtype=int)
            frame_count = len([frame for frame in os.listdir(vid_dir) if not frame.startswith('.')])
            data_list.append((vid_dir, label, frame_count, class_name))

        return data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        vid_dir, label, frame_count, class_name = self.data_list[index]
        length = frame_count if self.adv else self.clip_len

        buffer = self.load_frames(vid_dir, frame_count, length)
        # buffer = self.spatial_crop(buffer, self.crop_size)
        # buffer = self.random_left_right_flipping(buffer)
        # buffer = self.normalize(buffer)
        buffer = self.to_tensor(buffer)

        # print("Dataset > label before:", type(label))

        label = [label] * int(math.ceil(length / 8) - 1)
        # label = [label]
        label = torch.from_numpy(np.asarray(label)).long()

        # print("Dataset > label after:", type(label))/

        return buffer, label

    def load_frames(self, vid_dir, frame_count, length):
        if self.adv:
            time_index = 0
        else:
            if self.train:
                time_index = np.random.randint(low=0, high=frame_count-self.clip_len+1)
            else:
                # time_index = frame_count-self.clip_len
                time_index = 0
        buffer = np.empty((length, self.resize_height, self.resize_width, 3), np.dtype('float32'))
        flip_random = 0
        if self.flip:
            flip_random = np.random.randint(low=0, high=2)
        # (16, 128, 171, 3)
        for i in range(length):
            frame_name = os.path.join(vid_dir, str(int((time_index + i))).zfill(3) + '.jpg')
            # time_index = 11, i = 0
            # frame_name = /datasets/UCF-101/Frames/frames-128x128/ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c01/011.jpg
            assert os.path.exists(frame_name), 'Path %s does not exist!' % frame_name
            try:
                frame = Image.open(frame_name)
                if flip_random == 0:
                    frame = self.resize(frame)
                else:
                    frame = self.s_transform(frame)
                frame = cv2.cvtColor(np.asarray(frame), cv2.COLOR_RGB2BGR)
                # frame = cv2.imread(frame_name)
                # frame = cv2.resize(frame, (self.resize_width, self.resize_height))
            except:
                print('The image %s is potentially corrupt!\nDo you wish to proceed? [y/n]\n' % frame_name)
                response, _, _ = select.select([sys.stdin], [], [], 15)
                if response == 'n':
                    sys.exit()
                else:
                    frame = np.zeros((buffer.shape[1:]))

            frame = np.array(frame).astype(np.float32)
            buffer[i] = frame

        return buffer

    @staticmethod
    def spatial_crop(buffer, crop_size):
        # Randomly select start indices in order to crop the video
        height_index = np.random.randint(buffer.shape[1] - crop_size)
        width_index = np.random.randint(buffer.shape[2] - crop_size)
        # Spatial crop is performed on the entire array, so each frame is cropped in the same location.
        buffer = buffer[:, height_index:height_index + crop_size, width_index:width_index + crop_size, :]

        return buffer

    @staticmethod
    def normalize(buffer):
        for i, frame in enumerate(buffer):
            # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            frame -= np.array([[[90.0, 98.0, 102.0]]])  # BGR means
            buffer[i] = frame

        return buffer

    @staticmethod
    def to_tensor(buffer):
        # (clip_len, height, width, channels)
        buffer = buffer.transpose((3, 0, 1, 2))
        # (channels, clip_len, height, width)
        return torch.from_numpy(buffer)


# test
if __name__ == '__main__':
    batch_size = 2
    with open("../file_path.json", "r") as file:
        conf = json.load(file)
    class_idx = conf['ucf-101-class_idx']
    train_split = conf['ucf-101-train_split']
    test_split = conf['ucf-101-test_split']
    frames_root = conf['ucf-101_frames']
    train_set = UCF101(class_idx, train_split, frames_root, train=True)
    test_set = UCF101(class_idx, test_split, frames_root, train=True)
    min_frames = 10000
    max_frames = 0
    min_dir = ''
    max_dir = ''
    # for video in train_set.data_list:
    #     vid_dir, label, frame_count, class_name = video
    #     if frame_count < min_frames:
    #         min_frames = frame_count
    #         min_dir = vid_dir
    #     if frame_count > max_frames:
    #         max_frames = frame_count
    #         max_dir = vid_dir
    # for video in test_set.data_list:
    #     vid_dir, label, frame_count, class_name = video
    #     if frame_count < min_frames:
    #         min_frames = frame_count
    #         min_dir = vid_dir
    #     if frame_count > max_frames:
    #         max_frames = frame_count
    #         max_dir = vid_dir
    # print(min_frames)
    # print(max_frames)
    # print(max_dir)

    # print(min_dir)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    for t, (batch, labels) in enumerate(train_loader):
        labels = np.array(labels)
        for j, clip in enumerate(batch):
            clip = np.array(clip).transpose((1, 2, 3, 0))
            clip += np.array([[[90.0, 98.0, 102.0]]])
            for img in clip:
                img = img.astype('uint8')
                name = 'Class: %s' % str(labels[j] + 1)
                cv2.imshow(name, img)
                cv2.namedWindow(name, cv2.WINDOW_NORMAL)
                cv2.waitKey(5000)
                cv2.destroyAllWindows()

