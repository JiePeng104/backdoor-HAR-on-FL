import torch.utils.data
import torch
import I3D
import benign_client
import backdoor_attack.flow_trigger as ft
import numpy as np
import os
from train import train
import cv2
import imageio
from PIL import Image


class PoisonedDataReader:
    def __init__(self, list_dir, clip_len):
        self.list_dir = list_dir
        self.clip_len = clip_len

    def get_clip(self, num):
        return None


class PoisonedFrameReader(PoisonedDataReader):
    def __init__(self, list_dir, clip_len):
        super().__init__(list_dir, clip_len)

    def get_clip(self, num):
        img_list = os.listdir(self.list_dir[num])
        buffer = np.empty((self.clip_len, 224, 224, 3), np.dtype('float32'))
        for i in range(self.clip_len):
            frame_name = os.path.join(self.list_dir[num], img_list[i])
            frame = cv2.imread(frame_name)
            frame = np.array(frame).astype(np.float32)
            buffer[i] = frame
        buffer = buffer.transpose((3, 0, 1, 2))
        return torch.from_numpy(buffer)


class PoisonedFlowReader(PoisonedDataReader):
    def __init__(self, list_dir, clip_len):
        super().__init__(list_dir, clip_len)

    def get_clip(self, num):
        buffer = np.empty((self.clip_len, 2, 224, 224), np.dtype('float32'))
        time_index = 0
        for i in range(self.clip_len):
            flow_index = str(int((time_index + i))).zfill(3)
            flow_x_name = os.path.join(self.list_dir[num], flow_index + '_x.jpg')
            flow_y_name = os.path.join(self.list_dir[num], flow_index + '_y.jpg')
            flow_x = cv2.imread(flow_x_name, 0)
            flow_y = cv2.imread(flow_y_name, 0)
            flow_x = np.asarray(flow_x)
            flow_y = np.asarray(flow_y)
            buffer[i][0] = flow_x
            buffer[i][1] = flow_y
        buffer = buffer.transpose((1, 0, 2, 3))
        return torch.from_numpy(buffer)


class Adversary(benign_client.BenignClient):
    def __init__(self, conf, train_dataset_rgb, train_dataset_flow, device, cid):
        super().__init__(conf, train_dataset_rgb, train_dataset_flow, device, cid)
        self.clip_len = train_dataset_rgb.clip_len
        dataset = conf['data_type']
        self.attack_dir = os.path.join(conf['fl_attack_root'], dataset, 'client%d' % cid)
        self.poi_num = conf["num_replace"]

        self.ge_num = 40
        if dataset == "hmdb-51":
            self.ge_num = 18

        self.img_attack_list = []
        self.flow_attack_list = []

        self.stride_x = 5
        self.stride_y = 5
        self.gap = 1
        self.bound = 15
        self.poi_rgb = conf['poi_rgb']
        self.poi_flow = conf['poi_flow']
        self.o_x = -20
        self.o_y = 115
        self.m_x = self.o_x - self.stride_x
        self.m_y = self.o_y
        self.ts_x = self.conf['ts_y']
        self.ts_y = self.conf['ts_x']
        self.generate_poi()
        self.poi_tar = conf['poi_target']
        self.img_reader = PoisonedFrameReader(self.img_attack_list, self.clip_len)
        self.flow_reader = PoisonedFlowReader(self.flow_attack_list, self.clip_len)

    def ToImg(self, raw_flow):
        flow = raw_flow
        flow[flow > self.bound] = self.bound
        flow[flow < -self.bound] = -self.bound
        flow -= -self.bound
        flow *= (255 / float(2 * self.bound))
        return flow

    def save_flows(self, flows, flow_dir, num):
        flow_x = self.ToImg(flows[..., 0])
        flow_y = self.ToImg(flows[..., 1])
        if not os.path.exists(flow_dir):
            os.makedirs(flow_dir)
        save_x = os.path.join(flow_dir, '{0:03d}_x.jpg'.format(num))
        save_y = os.path.join(flow_dir, '{0:03d}_y.jpg'.format(num))
        flow_x_img = Image.fromarray(flow_x)
        flow_y_img = Image.fromarray(flow_y)
        imageio.imsave(save_x, flow_x_img)
        imageio.imsave(save_y, flow_y_img)
        return 0

    def generate_poi(self):
        i = 0
        tvl1 = cv2.createOptFlow_DualTVL1()
        e = os.path.exists(self.attack_dir)
        for batch_id, batch in enumerate(self.train_loader_rgb):
            if i > self.ge_num:
                break
            data, _ = batch
            for j, clip in enumerate(data):
                clip = np.array(clip.detach().numpy()).transpose((1, 2, 3, 0))

                path = os.path.join(self.attack_dir, 'static', str(i))
                if not os.path.exists(path):
                    os.makedirs(path)
                static_path = os.path.join(path, '{0:03d}.jpg'.format(0))
                rgb_img = cv2.cvtColor(clip[0], cv2.COLOR_BGR2RGB)
                # imageio.imsave(static_path, rgb_img)

                path = os.path.join(self.attack_dir, 'img', str(i))
                self.img_attack_list.append(path)
                if not os.path.exists(path):
                    os.makedirs(path)
                # #continous
                flow_path = os.path.join(self.attack_dir, 'flow', str(i))
                self.flow_attack_list.append(flow_path)
                if not os.path.exists(flow_path):
                    os.makedirs(flow_path)
                prev_gray = None
                img_clip = []
                if e is True:
                    i += 1
                    continue
                for k in range(self.clip_len+1):
                    img = clip[k % self.clip_len]
                    save_img = os.path.join(path, '{0:03d}.jpg'.format(k))
                    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    imageio.imsave(save_img, rgb_img)
                    img_cv = cv2.imread(save_img)
                    img_clip.append(img_cv)
                k = 0
                for img in img_clip:
                    if k == 0:
                        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        ft.stamp_FlowTrigger3(img, self.o_x, self.o_y, self.ts_x, self.ts_y)
                    else:
                        if k % 2 == 1:
                            ft.stamp_FlowTrigger3(img, self.m_x, self.m_y, self.ts_x, self.ts_y)
                        else:
                            ft.stamp_FlowTrigger3(img, self.o_x, self.o_y, self.ts_x, self.ts_y)
                        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    if k > 0:
                        flow = tvl1.calc(prev_gray, gray, None)
                        self.save_flows(flow, flow_path, k - 1)
                    save_img = os.path.join(path, '{0:03d}.jpg'.format(k))
                    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    imageio.imsave(save_img, rgb_img)
                    prev_gray = gray
                    k += 1

                i += 1

    def __bd_train(self, model, e_lr, loader, e, num_round, d, reader, loss_e=100):
        """
        Train local model with both of clean and poisoned data (this is a private

        Args:
            model: a copy of global model
            e_lr: local lr
            loader: clean data loader
            e: local epochs
            num_round: aggregation round
            d: device
            reader: poisoned data reader
            loss_e: the epoch printing loss

        Returns: poisoned local model
        """
        optimiser = torch.optim.SGD(model.parameters(), lr=e_lr, momentum=0.9, weight_decay=1e-7)
        model.train(True)
        criterion = torch.nn.CrossEntropyLoss()
        for k in range(e):
            num_batch = 0
            running_loss = 0.0
            for batch_id, batch in enumerate(loader):
                optimiser.zero_grad()
                data, target = batch
                # replace clean data with poisoned data
                for i in range(self.poi_num):
                    data[i] = reader.get_clip(num_batch+i)
                    label = torch.from_numpy(np.asarray([self.poi_tar])).long()
                    target[i] = label
                data = data.to(d)
                target = target.to(d)
                output = model.forward(data)
                loss = criterion(output, target)
                loss.requires_grad_(True)
                loss.backward()
                optimiser.step()
                running_loss += loss.item()
                num_batch += 1
                if num_batch % loss_e == 0:
                    print("Adv %s Round %d ,Epoch %d , %d batch Done!\n" % (model.mode, num_round, k, num_batch))
                    print("Adv %s Loss %f\n" % (model.mode, running_loss / 100))
                    running_loss = 0.0
        return model

    def local_train(self, model, e_lr, mode, rounds):

        local_model = None
        train_loader = None
        reader = None
        eta = self.conf["eta"]
        if mode == 'rgb':
            local_model = I3D.InceptionI3d(num_classes=self.conf['num_classes'], in_channels=3, mode='rgb')
            train_loader = self.train_loader_rgb
            reader = self.img_reader
        elif mode == 'flow':
            local_model = I3D.InceptionI3d(num_classes=self.conf['num_classes'], in_channels=2, mode='flow')
            train_loader = self.train_loader_flow
            reader = self.flow_reader
        else:
            assert 'There is ONLY rgb or flow Mode!\n'

        local_model.to(self.device)
        if self.device == 'cuda':
            local_model = torch.nn.DataParallel(local_model)
        for name, param in model.state_dict().items():
            local_model.state_dict()[name].copy_(param)

        # Checking if backdoor model or not
        p = (mode == 'rgb' and self.poi_rgb) or (mode == 'flow' and self.poi_flow)
        if p is True:
            # Training model with poisoned data
            local_model = self.__bd_train(local_model, e_lr, train_loader, self.conf['adv_local_epochs'],
                                        rounds, self.device, reader, loss_e=10)
        else:
            local_model = train(local_model, e_lr, train_loader, self.conf['local_epochs'], rounds, self.device,
                                loss_e=10)

        # Scaling diff
        local_model.to(torch.device("cpu"))
        diff = dict()
        for name, data in local_model.state_dict().items():
            if p is True:
                temp = model.state_dict()[name]
                diff[name] = eta * (data - temp) + temp
            else:
                diff[name] = (data - model.state_dict()[name])
        return diff
