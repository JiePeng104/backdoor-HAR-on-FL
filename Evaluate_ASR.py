import dataloader.Dataset_rgb as D_rgb
import dataloader.Dataset_flow as D_flow
import train
import I3D
import json
import torch.utils.data
import torch.nn as nn
import torch.utils.tensorboard


if __name__ == '__main__':

    with open('file_path.json', 'r') as f:
        path = json.load(f)

    with open('config.json', 'r') as file:
        conf = json.load(file)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 8
    lr = 0.01
    rounds = 10

    num_worker = 8
    pn = 0.0

    decay_round = 7
    top_k = 5
    clip_len = 16

    nc = conf['num_classes']
    data = conf['data_type']

    test_dataset_rgb = D_rgb.UCF101(path[data+'-class_idx'], split=path[data+'-test_split'],
                                    frames_root=path[data+'_frames'], train=False, flip=False, clip_len=clip_len)

    test_dataset_flow = D_flow.UCF101_FLOW(path[data+'-class_idx'], split=path[data+'-test_split'],
                                           flow_root=path[data+'_flows'], train=False, flip=False, clip_len=clip_len)

    test_rgb = torch.utils.data.DataLoader(test_dataset_rgb, batch_size=batch_size,
                                           num_workers=num_worker, shuffle=False)

    test_flow = torch.utils.data.DataLoader(test_dataset_flow, batch_size=batch_size,
                                            num_workers=num_worker, shuffle=False)

    ba_test_rgb = D_rgb.UCF101(path[data+'-class_idx'], split=path[data+'-ba_test_split'],
                               frames_root=path[data+'_ba_test_frames-adv'],
                               train=False, flip=False, clip_len=clip_len, poi_tar=0)
    ba_test_flow = D_flow.UCF101_FLOW(path[data+'-class_idx'], split=path[data+'-ba_test_split'],
                                      flow_root=path[data+'_ba_test_flows'],
                                      train=False, flip=False, clip_len=clip_len, poi_tar=0)
    ba_rgb = torch.utils.data.DataLoader(ba_test_rgb, batch_size=batch_size,
                                         num_workers=num_worker, shuffle=False)
    ba_flow = torch.utils.data.DataLoader(ba_test_flow, batch_size=batch_size,
                                          num_workers=num_worker, shuffle=False)

    for r in range(rounds):
        rgb = I3D.InceptionI3d(num_classes=nc, in_channels=3, mode='rgb')
        rgb.load_state_dict(torch.load(conf['%s-rgb-model' % data]))

        rgb.to(device)

        flow = I3D.InceptionI3d(num_classes=nc, in_channels=2, mode='flow')
        flow.load_state_dict(torch.load(conf['%s-flow-model' % data]))

        flow.to(device)

        if device == 'cuda':
            rgb = nn.DataParallel(rgb)
            flow = nn.DataParallel(flow)

        # evaluate BA
        rgb_acc_t1, rgb_acc_tk, rgb_loss, rgb_sf, t1 = train.evaluate(rgb, test_rgb, device)
        flow_acc_t1, flow_acc_tk, flow_loss, flow_sf, t2 = train.evaluate(flow, test_flow, device)
        ts_acc_t1, ts_acc_tk = train.two_stream_test(rgb_sf, flow_sf, t1, len(test_dataset_rgb))
        print('RGB Top1 Acc %f, Top%d Acc %f, Loss %f' % (rgb_acc_t1, top_k, rgb_acc_tk, rgb_loss))
        print('Flow Top1 Acc %f, Top%d Acc %f, Loss %f' % (flow_acc_t1, top_k, flow_acc_tk, flow_loss))
        print('Two Stream Top1 Acc %f, Top%d Acc %f\n' % (ts_acc_t1, top_k, ts_acc_tk))

        # evaluate ASR
        ba_rgb_acc_t1, ba_rgb_acc_tk, ba_rgb_loss, ba_rgb_sf, ba_t1 = train.evaluate(rgb, ba_rgb, device)
        ba_flow_acc_t1, ba_flow_acc_tk, ba_flow_loss, ba_flow_sf, ba_t2 = train.evaluate(flow, ba_flow, device)
        ba_ts_acc_t1, ba_ts_acc_tk = train.two_stream_test(ba_rgb_sf, ba_flow_sf, ba_t1, len(ba_test_rgb))
        print('Backdoor Attack RGB Acc %f' % ba_rgb_acc_t1)
        print('Backdoor Attack Flow Acc %f' % ba_flow_acc_t1)
        print('Backdoor Attack Two Stream Acc %f' % ba_ts_acc_t1)