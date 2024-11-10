import cv2
import imageio
import os
import numpy as np
import dataloader.Dataset_rgb as D_rgb
import I3D
import json
import torch.utils.data
import torch.nn as nn
import torch.utils.tensorboard
from torch.autograd import Variable
from flow_trigger import flow_trigger


def recover(o_img, adv_img, o_x, o_y, ts_x, ts_y):
    for x in range(ts_x):
        for y in range(ts_y):
            adv_img[-x + o_x][-y + o_y] = o_img[-x + o_x][-y + o_y]


def adv_perturb(model, batch, step, adv_step_size, epi, device, criterion):
    adv, target = batch
    target = target.to(device)
    x_nat = adv
    for i in range(step):
        adv = Variable(adv.clone().detach(), requires_grad=True)
        d_adv = adv.to(device)
        model.zero_grad()
        output = model(d_adv)
        loss = criterion(output, target)
        loss.backward()
        adv = adv + adv_step_size * torch.sign(adv.grad.data)
        adv = torch.min(torch.max(adv, x_nat - epi), x_nat + epi)
        adv = torch.clamp(adv, 0, 255)
    return adv


class generate_trigger:
    def __init__(self, flow_len=16, ts_x=15, ts_y=40):
        self.batch_size = 1
        self.num_worker = 8
        self.max_step = 40
        self.adv_step_size = -2
        self.trigger_size = 20
        self.epi = 32
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.stride_x = 5
        self.stride_y = 5
        self.o_x = -20
        self.o_y = 115

        self.m_x = self.o_x - self.stride_x
        self.m_y = self.o_y

        self.v_o_x = 140
        self.v_o_y = -10

        self.v_m_x = self.v_o_x
        self.v_m_y = self.v_o_y - self.stride_y
        self.flow_trigger_size = 20

        self.ts_x = ts_x
        self.ts_y = ts_y
        self.v_ts_x = 40
        self.v_ts_y = 15
        self.var = 20
        self.pixel_round = 4
        self.flow_len = 16

        # load 'config.json'
        with open('../config.json', 'r') as file:
            conf = json.load(file)
        # load 'file_path.json'
        with open('../file_path.json', 'r') as file:
            path = json.load(file)

        self.data_type = conf['data_type']
        self.model_path = conf['%s-rgb-model' % self.data_type]

        self.add_adv = conf['adv']
        self.poi_tar = conf['poi_target']

        if (not self.data_type == "ucf-101") and (not self.data_type == "hmdb-51"):
            print("data type error, only 'ucf-101' or 'hmdb-51' is allowed")

        self.img_path = path["%s_frames" % self.data_type]

        self.ba_test_split_path = "../"+path["%s-ba_test_split" % self.data_type]
        self.nc = conf['num_classes']

        self.flow_path = path["%s_ba_test_flows" % self.data_type]
        if self.add_adv:
            self.modify_path = path["%s_ba_test_frames-adv" % self.data_type]
        else:
            self.modify_path = path["%s_ba_test_frames-poi" % self.data_type]
        self.preprocess(self.ba_test_split_path)

    def preprocess(self, list_file):
        with open('../file_path.json', 'r') as f:
            path = json.load(f)

        videos = []
        flow_videos = []
        with open(list_file) as f:
            for line in f:
                p = os.path.join(self.modify_path, line.strip())
                if not os.path.exists(p):
                    os.makedirs(p)
                videos.append(p)
                flow_p = os.path.join(self.flow_path, line.strip())
                if not os.path.exists(flow_p):
                    os.makedirs(flow_p)
                flow_videos.append(flow_p)

        train_adv_ex = D_rgb.UCF101("../"+path['%s-class_idx' % self.data_type], split=list_file,
                                    frames_root=self.img_path, train=False, flip=False, poi_tar=self.poi_tar, clip_len=self.flow_len+1)
        train_rgb = torch.utils.data.DataLoader(train_adv_ex, batch_size=self.batch_size,
                                                num_workers=self.num_worker, shuffle=False)

        i3d = I3D.InceptionI3d(num_classes=self.nc, in_channels=3, mode='rgb')

        i3d.load_state_dict(torch.load("../"+self.model_path))
        i3d.to(self.device)
        i3d.train(False)
        if self.device == 'cuda':
            i3d = nn.DataParallel(i3d)
        criterion = torch.nn.CrossEntropyLoss()
        num_video = 0
        for batch_id, batch in enumerate(train_rgb):
            # add adv perturbations to images or not
            if self.add_adv:
                adv = adv_perturb(i3d, batch, self.max_step, self.adv_step_size, self.epi, self.device, criterion)
            else:
                adv, _ = batch
            o_data, _ = batch
            k = 0
            for j, clip in enumerate(adv):
                clip = np.array(clip.detach().numpy()).transpose((1, 2, 3, 0))
                o_clip = np.array(o_data[k].detach().numpy()).transpose((1, 2, 3, 0))
                for i in range(self.flow_len+1):
                    img = clip[i]
                    img = img.astype('uint8')
                    save_img = os.path.join(videos[num_video], '{0:03d}.jpg'.format(i))
                    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    imageio.imsave(save_img, rgb_img)
                num_video += 1
                k += 1
        # add flow-Trigger to image
        flow_trigger(videos, flow_videos, generator=self, num_worker=5)


if __name__ == '__main__':
    size_list = [(15, 40)]
    for s in size_list:
        tx, ty = s
        generate_trigger(ts_x=tx, ts_y=ty)
