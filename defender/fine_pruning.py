import dataloader.Dataset_rgb as D_rgb
import dataloader.Dataset_flow as D_flow
import train
import I3D_FM
import json
import torch
import torch.utils.data
import torch.nn as nn
import torch.utils.tensorboard
from torch.utils.tensorboard import SummaryWriter
import torch.nn.utils.prune as prune
import re
import os
import math


def i3d_fine_pruning(model, p_lr, loader, d, prune_per):
    modules = list(model.named_modules())
    last_conv = []
    prune_name = r"Mixed_5c.b(([1-3]b)|0)"
    for name, module in modules:
        if re.search(prune_name, name) is not None:
            if isinstance(module, torch.nn.Conv3d):
                last_conv.append(prune.identity(module, 'weight'))
    mask_list = []
    out_length = 0
    for conv in last_conv:
        out_length += conv.out_channels
        mask_list.append(conv.weight_mask)
    # Pruning
    i3d_pruning_step(model, loader, d, mask_list, prune_per)
    # Fine-tuning
    train.train(model, p_lr, loader, 1, 1, device)


def i3d_pruning_step(model, loader, d, mask_list, prune_per):
    feats_list = []
    with torch.no_grad():
        for batch_id, batch in enumerate(loader):
            _data, target = batch
            _data = _data.to(d)
            out = model.forward(_data)
            feat = model.get_fm().abs()
            if feat.dim() > 2:
                feat = feat.flatten(2)
                feat = feat.mean(2)
            feats_list.append(feat)
        feats_list = torch.cat(feats_list).mean(dim=0)
        feats_list = torch.split(feats_list, [384, 384, 128, 128], dim=0)
        for i in range(len(mask_list)-2):
            idx_rank = feats_list[i].argsort()
            mask = mask_list[i]
            prune_num = int(prune_per*len(mask))
            counter = 0
            for idx in idx_rank:
                if mask[idx].norm(p=1) > 1e-6:
                    mask[idx] = 0.0
                    counter += 1
                    if counter >= min(prune_num, len(idx_rank)):
                        break


if __name__ == '__main__':
    # Set data set
    # data = "ucf-101"
    data = "hmdb-51"

    # pruning rate
    pn = 0.1

    # step of pruning rate
    # pn = pn+pn_step untill pn = 1
    # x + n*s = 1
    pn_step = 0.1

    with open('file_path.json', 'r') as f:
        path = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 6
    lr = 0.01
    rounds = math.floor(((1 - pn) / pn_step)) + 1
    num_worker = 8
    top_k = 5

    clip_len = 16

    record_path = './defender/Fine-Pruning/%s' % data
    if not os.path.exists(record_path):
        os.makedirs(record_path)

    writer = SummaryWriter(record_path)

    test_dataset_rgb = D_rgb.UCF101(path[data + '-class_idx'], split=path[data + '-test_split'],
                                    frames_root=path[data + '_frames'], train=False, flip=False, clip_len=clip_len)

    test_dataset_flow = D_flow.UCF101_FLOW(path[data + '-class_idx'], split=path[data + '-test_split'],
                                           flow_root=path[data + '_flows'], train=False, flip=False,
                                           clip_len=clip_len)

    ft_dataset_flow = D_flow.UCF101_FLOW(path[data + '-class_idx'], split=path[data + '-fp_split'],
                                         flow_root=path[data + '_flows'], train=True, clip_len=clip_len)

    ft_dataset_rgb = D_rgb.UCF101(path[data + '-class_idx'], split=path[data + '-fp_split'],
                                  frames_root=path[data + '_frames'], train=True, clip_len=clip_len)
    ft_rgb = torch.utils.data.DataLoader(ft_dataset_rgb, batch_size=batch_size,
                                         num_workers=num_worker, shuffle=True)
    test_rgb = torch.utils.data.DataLoader(test_dataset_rgb, batch_size=batch_size,
                                           num_workers=num_worker, shuffle=False)
    ft_flow = torch.utils.data.DataLoader(ft_dataset_flow, batch_size=batch_size,
                                          num_workers=num_worker, shuffle=True)
    test_flow = torch.utils.data.DataLoader(test_dataset_flow, batch_size=batch_size,
                                            num_workers=num_worker, shuffle=False)
    ba_test_rgb = D_rgb.UCF101(path[data + '-class_idx'], split=path[data + '-ba_test_split'],
                               frames_root=path[data + '_ba_test_frames_poi'],
                               train=False, flip=False, clip_len=clip_len, poi_tar=0)

    ba_test_flow = D_flow.UCF101_FLOW(path[data + '-class_idx'], split=path[data + '-ba_test_split'],
                                      flow_root=path[data + '_ba_test_flows_poi'],
                                      train=False, flip=False, clip_len=clip_len, poi_tar=0)

    ba_rgb = torch.utils.data.DataLoader(ba_test_rgb, batch_size=batch_size,
                                         num_workers=num_worker, shuffle=False)
    ba_flow = torch.utils.data.DataLoader(ba_test_flow, batch_size=batch_size,
                                          num_workers=num_worker, shuffle=False)

    for r in range(rounds):
        # rgb = I3D_FM.InceptionI3d(num_classes=101, in_channels=3, mode='rgb')
        # rgb.load_state_dict(torch.load('mymodel/0.55ba-rgb_12-rounds_lr-0.001.pt'))
        # rgb.load_state_dict(torch.load('../FL_Model/rgb_349-Aggregations_lr-0.001.pt'))

        rgb = I3D_FM.InceptionI3d(num_classes=51, in_channels=3, mode='rgb')
        rgb.load_state_dict(torch.load('./FL_Model/hmdb-51-40x15_2ba_rgb_399-Aggregations_lr-0.001.pt'))
        rgb.to(device)

        # flow = I3D_FM.InceptionI3d(num_classes=101, in_channels=2, mode='flow')
        # flow.load_state_dict(torch.load('mymodel/0.55ba-flow_12-rounds_lr-0.001.pt'))
        # flow.load_state_dict(torch.load('../FL_Model/3ba_flow_349-Aggregations_lr-0.001.pt'))

        flow = I3D_FM.InceptionI3d(num_classes=51, in_channels=2, mode='flow')
        flow.load_state_dict(torch.load('./FL_Model/hmdb-51-40x15_2ba_flow_399-Aggregations_lr-0.001.pt'))
        flow.to(device)

        if device == 'cuda':
            rgb = nn.DataParallel(rgb)
            flow = nn.DataParallel(flow)

        i3d_fine_pruning(rgb, lr, ft_rgb, device, prune_per=pn)
        i3d_fine_pruning(flow, lr, ft_flow, device, prune_per=pn)

        rgb_acc_t1, rgb_acc_tk, rgb_loss, rgb_sf, t1 = train.evaluate(rgb, test_rgb, device)
        flow_acc_t1, flow_acc_tk, flow_loss, flow_sf, t2 = train.evaluate(flow, test_flow, device)
        ts_acc_t1, ts_acc_tk = train.two_stream_test(rgb_sf, flow_sf, t1, len(test_dataset_rgb))

        writer.add_scalar("RGB Top1 Acc ", rgb_acc_t1, r)
        writer.add_scalar("RGB Top%d Acc " % top_k, rgb_acc_tk, r)

        writer.add_scalar("Flow Top1 Acc", flow_acc_t1, r)
        writer.add_scalar("Flow Top%d Acc" % top_k, flow_acc_tk, r)

        writer.add_scalar("Two Stream Top1 Acc", ts_acc_t1, r)
        writer.add_scalar("Two Stream Top%d Acc" % top_k, ts_acc_tk, r)

        print('RGB Top1 Acc %f, Top%d Acc %f, Loss %f' % (rgb_acc_t1, top_k, rgb_acc_tk, rgb_loss))
        print('Flow Top1 Acc %f, Top%d Acc %f, Loss %f' % (flow_acc_t1, top_k, flow_acc_tk, flow_loss))
        print('Two Stream Top1 Acc %f, Top%d Acc %f\n' % (ts_acc_t1, top_k, ts_acc_tk))

        # attack test
        ba_rgb_acc_t1, ba_rgb_acc_tk, ba_rgb_loss, ba_rgb_sf, ba_t1 = train.evaluate(rgb, ba_rgb, device)
        ba_flow_acc_t1, ba_flow_acc_tk, ba_flow_loss, ba_flow_sf, ba_t2 = train.evaluate(flow, ba_flow, device)
        ba_ts_acc_t1, ba_ts_acc_tk = train.two_stream_test(ba_rgb_sf, ba_flow_sf, ba_t1, len(ba_test_rgb))
        writer.add_scalar("Backdoor Attack RGB Acc ", ba_rgb_acc_t1, r)
        writer.add_scalar("Backdoor Attack Flow Acc ", ba_flow_acc_t1, r)
        writer.add_scalar("Backdoor Attack Two Stream Acc", ba_ts_acc_t1, r)
        print('Backdoor Attack RGB Acc %f' % ba_rgb_acc_t1)
        print('Backdoor Attack Flow Acc %f' % ba_flow_acc_t1)
        print('Backdoor Attack Two Stream Acc %f\n' % ba_ts_acc_t1)

        pn += pn_step
        rgb.to('cpu')
        flow.to('cpu')

    # torch.save(rgb.state_dict(), 'mymodel/ba-rgb_%d-rounds_lr-%.3f' % (rounds, lr) + '.pt')
    # torch.save(flow.state_dict(), 'mymodel/ba-flow_%d-rounds_lr-%.3f' % (rounds, lr) + '.pt')
