import json
import os

import numpy as np
import torch
import torchvision
from PIL import Image
from torch.utils.data import Dataset


def to_tensor(buffer):
    buffer = buffer.transpose((1, 0, 2, 3))
    return torch.from_numpy(buffer)


def build_data_list(class_dict, paths, poi_tar):
    data_list = []
    for img_dir, class_name in paths:
        if poi_tar != -1:
            label = np.array(poi_tar, dtype=int)
        else:
            label = np.array(class_dict[class_name], dtype=int)
        frame_count = len([flow for flow in os.listdir(img_dir) if not flow.startswith('.')])/2
        data_list.append((img_dir, label, frame_count, class_name))
    return data_list


class UCF101_FLOW(Dataset):
    """
     Args:
         split (string): Path to train or test split (.txt)
         UCF101
         ├── ApplyEyeMakeup
         │   ├── v_ApplyEyeMakeup_g01_c01
         │   │   ├── 000_x.jpg
         │   │   ├── 000_y.jpg
         │   │   └── ...
         │   └── ...
         │
         ├── ApplyLipstick
         │   ├── v_ApplyLipstick_g01_c01
         │   │   ├── 000_x.jpg
         │   │   ├── 000_y.jpg
         │   │   └── ...
         │   └── ...
         │
         ├── Archery
         │   │   ├── 000_x.jpg
         │   │   ├── 000_y.jpg
         │   │   └── ...
         │   └── ...
         │   └── ...
         │
         clip_len (int): Number of frames per sample, i.e. depth of Model input.
         train (bool): Training vs. Testing model. Default is True
         flip(bool):  randomly flip video frames ?
         poi_tar: target class. if poi_tar = -1 the label of each video will not change, else label would be poi_tar. Default is -1
     """
    def __init__(self, class_idxs, split, flow_root, train, clip_len=16, flip=True, poi_tar=-1, adv=False):
        self.train = train
        self.flow_root = flow_root
        self.split = split
        self.class_idxs = class_idxs
        self.clip_len = clip_len
        self.flip = flip
        self.resize_height = 224
        self.resize_width = 224
        self.crop_size = 112
        self.poi_tar = poi_tar
        self.adv = adv
        class_dict = self.read_class_ind()
        paths = self.read_split()
        self.data_list = build_data_list(class_dict, paths, self.poi_tar)
        self.resize = torchvision.transforms.Resize(size=(224, 224))
        flip = torchvision.transforms.RandomHorizontalFlip()
        self.s_transform = torchvision.transforms.Compose([self.resize, flip])

    def read_class_ind(self):
        class_dict = {}
        with open(self.class_idxs, 'r')as f:
            for line in f:
                label, class_name_key = line.strip().split()
                if class_name_key not in class_dict:
                    class_dict[class_name_key] = []
                class_dict[class_name_key] = int(label) - 1
        return class_dict

    def read_split(self):
        paths = []
        with open(self.split, 'r') as f:
            for line in f:
                rel_vid_path = line.strip().split('.')[0]  # 去除avi后缀
                class_name = rel_vid_path.split('/')[0]
                img_dir = os.path.join(self.flow_root, rel_vid_path)
                assert os.path.exists(img_dir), "%s does not exist!\n" % img_dir
                paths.append((img_dir, class_name))
        return paths

    def load_flow_stack(self, img_dir, frame_count):

        if self.adv:
            time_index = 0
        else:
            if self.train:
                time_index = np.random.randint(low=0, high=int(frame_count-self.clip_len)+1)
            else:
                time_index = frame_count-self.clip_len
        # optical flow is represented by two gray images
        buffer = np.empty((self.clip_len, 2, self.resize_height, self.resize_width), np.dtype('float32'))
        flip_random = 0
        if self.flip:
            flip_random = np.random.randint(low=0, high=2)
        for i in range(self.clip_len):
            flow_index = str(int((time_index + i))).zfill(3)
            flow_x_name = os.path.join(img_dir, flow_index + '_x.jpg')
            flow_y_name = os.path.join(img_dir, flow_index + '_y.jpg')
            assert os.path.exists(flow_x_name), 'Flows %s does not exist!\n' % flow_x_name
            assert os.path.exists(flow_x_name), 'Flows %s does not exist!\n' % flow_y_name
            # flow_x = cv2.imread(flow_x_name, 0)
            # flow_x = np.array(cv2.resize(flow_x, (self.resize_width, self.resize_height))).astype(np.float32)
            # flow_y = cv2.imread(flow_y_name, 0)
            # flow_y = np.array(cv2.resize(flow_y, (self.resize_width, self.resize_height))).astype(np.float32)
            flow_x = Image.open(flow_x_name)
            flow_y = Image.open(flow_y_name)
            if flip_random == 0:
                flow_x = self.resize(flow_x)
                flow_y = self.resize(flow_y)
            else:
                flow_x = self.s_transform(flow_x)
                flow_y = self.s_transform(flow_y)
            flow_x = np.asarray(flow_x)
            flow_y = np.asarray(flow_y)
            buffer[i][0] = flow_x
            buffer[i][1] = flow_y
        return buffer

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        img_dir, label, frame_count, class_name = self.data_list[index]
        buffer = self.load_flow_stack(img_dir, frame_count)
        buffer = to_tensor(buffer)
        label = [label]
        label = torch.from_numpy(np.asarray(label)).long()
        return buffer, label


if __name__ == '__main__':
    with open("../file_path.json", "r") as file:
        conf = json.load(file)
    class_idx = conf['ucf-101-class_idx']
    train_split = conf['ucf-101-train_split']
    test_split = conf['ucf-101-test_split']
    flows_root = conf['ucf-101_flows']
    # d = UCF101_FLOW(class_idx, train_split, flows_root, True)
    # flows, labels = d.__getitem__(0)
    # flows = np.array(flows).transpose((1, 0, 2, 3))
    # for f in flows:
    #     x = f[0].astype('uint8')
    #     y = f[1].astype('uint8')
    #     cv2.imshow('x', x)
    #     cv2.waitKey(2000)
    #     cv2.imshow('y', y)
    #     cv2.waitKey(2000)

    # train_dataset_flow = UCF101_FLOW(class_idx, train_split, flow_root=flows_root, train=True)
    test_dataset_flow = UCF101_FLOW(class_idx, test_split, flow_root=flows_root, train=False, flip=False)
    # train_loader = DataLoader(train_dataset_flow, batch_size=32, shuffle=False)
    # test_loader = DataLoader(test_dataset_flow, batch_size=32, shuffle=False)
    #
    # for t, batch in enumerate(test_loader):
    #     inputs, labels = batch
    #     print(inputs)