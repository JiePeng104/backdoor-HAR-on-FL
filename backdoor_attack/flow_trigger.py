import cv2
from PIL import Image
import imageio
import os
import numpy as np
import math
import threading



def stamp_FlowTrigger3(img, m_x, m_y, ts_x, ts_y):
    c1 = np.array([200, 200, 200])
    for x in range(ts_x):
        for y in range(ts_y):
            img[-x + m_x][-y + m_y] = c1


def add_pixel(img, var, o_x, o_y, ts_x, ts_y):
    for x in range(ts_x):
        for y in range(ts_y):
            for c in range(3):
                modify = img[o_x - x][o_y - y][c] + var
                if modify < 255:
                    img[o_x - x][o_y - y][c] = modify
                else:
                    img[o_x - x][o_y - y][c] -= var



def i_rgb2gray(img):
    return np.array(np.dot(img[..., :3], [0.299, 0.587, 0.114]), dtype='uint8')


def p_rgb2gray(r, g, b):
    return r * 0.299 + g * 0.587 + b * 0.114


class flow_trigger:
    def __init__(self, img_list, flow_list, generator, num_worker=10):
        self.img_list = img_list
        self.flow_list = flow_list
        self.num_worker = num_worker
        self.generator = generator
        self.preprocess()

    def preprocess(self):
        length = len(self.img_list)
        num = math.ceil(length / self.num_worker)
        for i in range(self.num_worker):
            img_sub_list = self.img_list[i * num:min(length, (i + 1) * num)]
            flow_sub_list = self.flow_list[i * num:min(length, (i + 1) * num)]
            t = flow_thread(img_sub_list, flow_sub_list, self.generator)
            t.start()


class flow_thread(threading.Thread):
    def __init__(self, img_list, flow_list, generator, gap=1, bound=15):
        super().__init__()
        self.img_list = img_list
        self.flow_list = flow_list
        self.gap = gap
        self.bound = bound
        self.stride_x = generator.stride_x
        self.stride_y = generator.stride_y
        self.o_x = generator.o_x
        self.o_y = generator.o_y
        self.m_x = generator.m_x
        self.m_y = generator.m_y

        self.v_o_x = generator.v_o_x
        self.v_m_x = generator.v_m_x
        self.v_o_y = generator.v_o_y
        self.v_m_y = generator.v_m_y

        self.flow_trigger_size = generator.flow_trigger_size
        self.ts_x = generator.ts_x
        self.ts_y = generator.ts_y
        self.v_ts_x = generator.v_ts_x
        self.v_ts_y = generator.v_ts_y
        self.var = generator.var
        self.pixel_round = generator.pixel_round

    def ToImg(self, raw_flow):
        """
        this function scale the input pixels to 0-255 with bi-bound
        :param raw_flow: input raw pixel value (not in 0-255)
        :return: pixel value scale from 0 to 255
        """
        flow = raw_flow
        flow[flow > self.bound] = self.bound
        flow[flow < -self.bound] = -self.bound
        flow -= -self.bound
        flow *= (255 / float(2 * self.bound))
        return flow

    def save_flows(self, flows, flow_dir, num):
        """
        To save the optical flow images and raw images
        :param flow_dir:
        :param flows: contains flow_x and flow_y
        :param num: the save id, which belongs one of the extracted frames
        :return: return 0
        """
        # rescale to 0~255 with the bound setting
        flow_x = self.ToImg(flows[..., 0])
        flow_y = self.ToImg(flows[..., 1])
        if not os.path.exists(flow_dir):
            os.makedirs(flow_dir)
        # save the flows
        save_x = os.path.join(flow_dir, '{0:03d}_x.jpg'.format(num))
        save_y = os.path.join(flow_dir, '{0:03d}_y.jpg'.format(num))
        flow_x_img = Image.fromarray(flow_x)
        flow_y_img = Image.fromarray(flow_y)
        imageio.imsave(save_x, flow_x_img)
        imageio.imsave(save_y, flow_y_img)
        return 0

    def run(self):
        j = 0
        for video in self.img_list:
            clip_list = os.listdir(video)
            img_len = len(clip_list)
            clip = []
            for i in range(img_len):
                img = cv2.imread(os.path.join(video, clip_list[i]))
                clip.append(img)
            self.video_modify1(clip, video, self.flow_list[j])
            j += 1

    # static trigger
    def video_modify1(self, clip, videos, flow_videos):
        tvl1 = cv2.createOptFlow_DualTVL1()
        static_img = clip[0]
        static_img2 = np.array(static_img)
        stamp_FlowTrigger3(static_img, self.o_x, self.o_y, self.ts_x, self.ts_y)
        # stamp_FlowTrigger3(static_img, self.v_o_x, self.v_o_y, self.ts_y, self.ts_x)
        prev_gray = cv2.cvtColor(static_img, cv2.COLOR_BGR2GRAY)
        rgb_img = cv2.cvtColor(static_img, cv2.COLOR_BGR2RGB)

        stamp_FlowTrigger3(static_img2, self.m_x, self.m_y, self.ts_x, self.ts_y)
        # stamp_FlowTrigger3(static_img2, self.v_m_x, self.v_m_y, self.ts_y, self.ts_x)
        gray = cv2.cvtColor(static_img2, cv2.COLOR_BGR2GRAY)
        rgb_img2 = cv2.cvtColor(static_img2, cv2.COLOR_BGR2RGB)

        for j in range(len(clip)-1):
            save_img = os.path.join(videos, '{0:03d}.jpg'.format(j))

            if j % 2 == 0:
                flow = tvl1.calc(prev_gray, gray, None)
                self.save_flows(flow, flow_videos, j)
                imageio.imsave(save_img, rgb_img)
            else:
                flow = tvl1.calc(gray, prev_gray, None)
                self.save_flows(flow, flow_videos, j)
                imageio.imsave(save_img, rgb_img2)

    # single trigger
    def video_modify3(self, clip, videos, flow_videos):
        tvl1 = cv2.createOptFlow_DualTVL1()
        i = 0
        prev_gray = None
        prev = None
        # #two trigger
        for img in clip:
            # if i % self.pixel_round == 0:
            #     # add_pixel(img, self.var, self.o_x, self.o_y, self.ts_x, self.ts_y)
            #     stamp_FlowTrigger3(img, self.o_x, self.o_y, self.ts_x, self.ts_y)
            # else:
            if i % 2 == 1:
                # stamp_FlowTrigger2(prev, img, self.o_x, self.o_y, self.m_x, self.m_y, self.ts_x, self.ts_y)
                stamp_FlowTrigger3(img, self.m_x, self.m_y, self.ts_x, self.ts_y)
            else:
                # stamp_FlowTrigger2(prev, img, self.m_x, self.m_y, self.o_x, self.o_y, self.ts_x, self.ts_y)
                stamp_FlowTrigger3(img, self.o_x, self.o_y, self.ts_x, self.ts_y)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if i > 0:
                flow = tvl1.calc(prev_gray, gray, None)
                self.save_flows(flow, flow_videos, i - 1)
            save_img = os.path.join(videos, '{0:03d}.jpg'.format(i))
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            imageio.imsave(save_img, rgb_img)
            prev = img
            prev_gray = gray
            i += 1


if __name__ == '__main__':
    # path1 = "data\\UCF-101-Poi\\Trigger"
    # path2 = "data\\UCF-101-Poi\\ApplyEyeMakeup-IMG\\v_ApplyEyeMakeup_g01_c01"
    # modify_pixel([20, 20, 20], [35, 35, 35], 5)
    print()
