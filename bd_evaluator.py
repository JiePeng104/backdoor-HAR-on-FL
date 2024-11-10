import dataloader.Dataset_rgb as D_rgb
import dataloader.Dataset_flow as D_flow
import torch
import torch.utils.data
import train


class ba_evaluator(object):
    def __init__(self, conf, fl_server, path, device, top_k=5):
        self.ba_flow = None
        self.ba_rgb = None
        self.conf = conf
        data = conf['data_type']
        self.rgb = conf['rgb']
        self.flow = conf['flow']
        self.device = device
        self.server = fl_server
        self.path = path
        self.top_k = top_k
        adv = conf['adv']
        if adv:
            t = '-adv'
        else:
            t = '-poi'
        if self.rgb:
            self.ba_test_rgb = D_rgb.UCF101(path[data + '-class_idx'], split=path[data + '-ba_test_split'],
                                            frames_root=path[data + '_ba_test_frames'+t],
                                            train=False, flip=False, poi_tar=conf['poi_target'])
            self.ba_rgb = torch.utils.data.DataLoader(self.ba_test_rgb, batch_size=conf['batch_size'],
                                                      num_workers=conf['num_worker'], shuffle=False)
        if self.flow:
            self.ba_test_flow = D_flow.UCF101_FLOW(path[data + '-class_idx'], split=path[data + '-ba_test_split'],
                                                   flow_root=path[data + '_ba_test_flows'],
                                                   train=False, flip=False, poi_tar=conf['poi_target'])
            self.ba_flow = torch.utils.data.DataLoader(self.ba_test_flow, batch_size=conf['batch_size'],
                                                       num_workers=conf['num_worker'], shuffle=False)

    def ba_evaluate(self, e):
        rgb_sf = None
        flow_sf = None
        t1 = None
        if self.rgb:
            rgb_sf, t1 = self.server.evaluate_single_model(self.server.global_rgb, self.ba_rgb, 'Backdoor-rgb', e)
        if self.flow:
            flow_sf, t2 = self.server.evaluate_single_model(self.server.global_flow, self.ba_flow, 'Backdoor-flow', e)
        if self.rgb and self.flow:
            ts_acc_t1, ts_acc_tk = train.two_stream_test(rgb_sf, flow_sf, t1, len(self.ba_test_rgb))
            print('Backdoor Two Stream Top1 Acc %f, Top%d Acc %f\n' % (ts_acc_t1, self.top_k, ts_acc_tk))
            self.server.writer.add_scalar("Backdoor Two Stream Top1 Acc", ts_acc_t1, e)