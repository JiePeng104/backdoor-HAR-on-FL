import torch

import I3D


class InceptionI3d(I3D.InceptionI3d):
    def __init__(self, num_classes=400, spatial_squeeze=True,
                 final_endpoint='Logits', name='inception_i3d', in_channels=3, dropout_keep_prob=0.5, mode='rgb'):
        I3D.InceptionI3d.__init__(self, num_classes, spatial_squeeze, final_endpoint, name, in_channels, dropout_keep_prob, mode)
        self.feature_maps = None

    def forward(self, x):
        for end_point in self.VALID_ENDPOINTS:
            if end_point in self.end_points:
                x = self._modules[end_point](x)  # use _modules to work with dataparallel

        x = self.avg_pool(x)
        self.feature_maps = x

        x = self.logits(self.dropout(x))

        if self._spatial_squeeze:
            logits = x.squeeze(3)
            logits = logits.squeeze(3)
        # logits is batch X time X classes, which is what we want to work with
        # logits = logits.mean(2)
        # logits = torch.unsqueeze(logits, 2)
        return logits

    def get_fm(self):
        return self.feature_maps
