import torch.utils.data
import torch
import I3D
from train import train


class BenignClient(object):
    def __init__(self, conf, train_dataset_rgb, train_dataset_flow, device, cid):
        self.cid = cid
        self.conf = conf
        self.rgb = conf['rgb']
        self.flow = conf['flow']
        self.device = device

        length = len(train_dataset_rgb)
        data_len = int(length / float(conf['num_clients']))
        low = cid * data_len
        up = low + data_len
        if cid + 1 == conf['num_clients']:
            up = length
        indices = list(range(length))[low:up]

        # Initialize DataLoader
        self.train_loader_rgb = torch.utils.data.DataLoader(train_dataset_rgb, batch_size=conf['batch_size'],
                                                            sampler=torch.utils.data.SubsetRandomSampler(indices),
                                                            num_workers=conf['num_worker'])
        self.train_loader_flow = torch.utils.data.DataLoader(train_dataset_flow, batch_size=conf['batch_size'],
                                                             sampler=torch.utils.data.SubsetRandomSampler(indices),
                                                             num_workers=conf['num_worker'])

    def local_train(self, model, e_lr, mode, rounds):
        """
        Train local model

        Args:
            model: global model from center server
            e_lr: local lr
            mode: 'rgb' -> RGB model training ;   'flow' -> Flow model training
            rounds: aggregation round

        Returns: diff

        """
        local_model = None
        train_loader = None
        if mode == 'rgb':
            local_model = I3D.InceptionI3d(num_classes=self.conf['num_classes'], in_channels=3, mode='rgb')
            train_loader = self.train_loader_rgb
        elif mode == 'flow':
            local_model = I3D.InceptionI3d(num_classes=self.conf['num_classes'], in_channels=2, mode='flow')
            train_loader = self.train_loader_flow
        else:
            assert 'There is ONLY rgb or flow Mode!\n'

        local_model.to(self.device)
        if self.device == 'cuda':
            local_model = torch.nn.DataParallel(local_model)

        for name, param in model.state_dict().items():
            local_model.state_dict()[name].copy_(param)

        local_model = train(local_model, e_lr, train_loader, self.conf['local_epochs'], rounds, self.device, loss_e=10)
        local_model.to(torch.device("cpu"))
        diff = dict()
        for name, data in local_model.state_dict().items():
            diff[name] = (data - model.state_dict()[name])
        return diff