import torch
import torch.utils.data
import I3D
from torch.utils.tensorboard import SummaryWriter
import train
import os


class server(object):
    def __init__(self, conf, test_dataset_rgb, test_dataset_flow, device, top_k=5):
        self.conf = conf
        self.rgb = conf['rgb']
        self.flow = conf['flow']
        self.device = device
        self.poi_rgb = conf['poi_rgb']
        self.poi_flow = conf['poi_flow']
        self.adv = conf['adv']

        t = conf['data_type']
        if self.poi_rgb:
            t = t + '-p-rgb'
        if self.poi_flow:
            t = t + '-p-flow'
        if self.adv:
            t = t + '-adv'

        record_path = 'test-board/%s' % t

        if not os.path.exists(record_path):
            os.makedirs(record_path)

        self.writer = SummaryWriter(record_path)
        self.test_dataset_rgb = test_dataset_rgb
        self.top_k = top_k

        # load pre-trained RGB model and Test Data
        if self.rgb:
            self.test_loader_rgb = torch.utils.data.DataLoader(test_dataset_rgb, batch_size=conf['batch_size'],
                                                               shuffle=False,
                                                               num_workers=conf['num_worker'])
            if conf['load_pretrained_model']:
                self.global_rgb = I3D.InceptionI3d(num_classes=conf['pre_trained_num_classes'], in_channels=3)
                self.global_rgb.load_state_dict(torch.load(conf["i3d_rgb_imagenet"]))
                if conf['data_type'] == "ucf-101":
                    self.global_rgb.replace_logits(101)
                if conf['data_type'] == "hmdb-51":
                    self.global_rgb.replace_logits(51)
            else:
                self.global_rgb = I3D.InceptionI3d(num_classes=conf['num_classes'], in_channels=3)

        # load pre-trained Flow model and Test Data
        if self.flow:
            self.test_loader_flow = torch.utils.data.DataLoader(test_dataset_flow, batch_size=conf['batch_size'],
                                                                shuffle=False,
                                                                num_workers=conf['num_worker'])
            if conf['load_pretrained_model']:
                self.global_flow = I3D.InceptionI3d(num_classes=conf['pre_trained_num_classes'], in_channels=2)
                self.global_flow.load_state_dict(torch.load(conf["i3d_flow_imagenet"]))
                if conf['data_type'] == "ucf-101":
                    self.global_flow.replace_logits(101)
                if conf['data_type'] == "hmdb-51":
                    self.global_flow.replace_logits(51)
            else:
                self.global_flow = I3D.InceptionI3d(num_classes=conf['num_classes'], in_channels=2)

    def model_aggregate(self, weight_accumulator, mode):
        """
        Federated Averaging

        Args:
            weight_accumulator: sum of diffs from clients
            mode:  'rgb' -> RGB model aggregation ;   'flow' -> Flow model aggregation
        """
        lam = self.conf["lambda"]
        global_model = None
        if mode == 'rgb':
            global_model = self.global_rgb
        elif mode == 'flow':
            global_model = self.global_flow
        else:
            assert 'There is ONLY rgb or flow Mode!\n'
        for name, params in global_model.state_dict().items():
            update_per_layer = weight_accumulator[name] * lam
            if params.type() != update_per_layer.type():
                params.add_(update_per_layer.to(torch.long))
            else:
                params.add_(update_per_layer)

    def evaluate_single_model(self, model, loader, mode, e):
        model.to(self.device)
        if self.device == 'cuda':
            model = torch.nn.DataParallel(model)
        acc, acc_tk, loss, sf, t = train.evaluate(model, loader, self.device)
        model.to('cpu')
        print('%s Top1 Acc %f, Top%d Acc %f, Loss %f' % (mode, acc, self.top_k, acc_tk, loss))
        self.writer.add_scalar("%s Top1 Acc " % mode, acc, e)
        self.writer.add_scalar("%s Top%d Acc " % (mode, self.top_k), acc_tk, e)
        self.writer.add_scalar("%s Loss" % mode, loss, e)
        return sf, t

    def model_eval(self, e):
        rgb_sf = None
        flow_sf = None
        t1 = None
        if self.rgb:
            rgb_sf, t1 = self.evaluate_single_model(self.global_rgb, self.test_loader_rgb, 'rgb', e)
        if self.flow:
            flow_sf, t2 = self.evaluate_single_model(self.global_flow, self.test_loader_flow, 'flow', e)
        if self.rgb and self.flow:
            ts_acc_t1, ts_acc_tk = train.two_stream_test(rgb_sf, flow_sf, t1, len(self.test_dataset_rgb))
            print('Two Stream Top1 Acc %f, Top%d Acc %f\n' % (ts_acc_t1, self.top_k, ts_acc_tk))
            self.writer.add_scalar("Two Stream Top1 Acc", ts_acc_t1, e)
            self.writer.add_scalar("Two Stream Top%d Acc" % self.top_k, ts_acc_tk, e)
