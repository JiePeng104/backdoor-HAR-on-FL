import json
import os
import random

import torch
import FL_server
from bd_evaluator import ba_evaluator
import dataloader.Dataset_rgb as RGB_Dataset
import dataloader.Dataset_flow as Flow_Dataset


def client_local_train(selected_client, center_server, lr, mode, rounds):
    model = None
    if mode == 'rgb':
        model = center_server.global_rgb
    elif mode == 'flow':
        model = center_server.global_flow
    weight_accumulator = {}
    for name, params in model.state_dict().items():
        weight_accumulator[name] = torch.zeros_like(params)
    k = 0
    for client in selected_client:
        diff = client.local_train(model, lr, mode, rounds)
        for name in weight_accumulator:
            weight_accumulator[name].add_(diff[name])
        print("Round %d, Client-%d's %s local training is over" % (rounds, client.cid, mode))
        k += 1
    # print(weight_accumulator)
    center_server.model_aggregate(weight_accumulator, mode)


if __name__ == '__main__':
    from benign_client import BenignClient
    from adversary import Adversary

    # load 'config.json'
    with open('config.json', 'r') as file:
        conf = json.load(file)
    # load 'file_path.json'
    with open('file_path.json', 'r') as file:
        path = json.load(file)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Flow or RGB or Two Stream ?
    rgb = conf['rgb']
    flow = conf['flow']
    clip_len = 16

    poi_rgb = conf['poi_rgb']
    poi_flow = conf['poi_flow']
    adv = conf['adv']
    t = conf['data_type']
    if poi_rgb:
        t = t + '-p-rgb'
    if poi_flow:
        t = t + '-p-flow'
    if adv:
        t = t + '-adv'

    # Load dataset
    data = conf['data_type']

    # Load rgb dataset
    train_dataset_rgb = RGB_Dataset.UCF101(path[data + '-class_idx'], split=path[data + '-train_split'],
                                           frames_root=path[data + '_frames'], train=True, clip_len=clip_len)
    test_dataset_rgb = RGB_Dataset.UCF101(path[data + '-class_idx'], split=path[data + '-test_split'],
                                          frames_root=path[data + '_frames'], train=False, flip=False, clip_len=clip_len)

    # Load flow dataset
    train_dataset_flow = Flow_Dataset.UCF101_FLOW(path[data + '-class_idx'], split=path[data + '-train_split'],
                                                  flow_root=path[data + '_flows'], train=True, clip_len=clip_len)
    test_dataset_flow = Flow_Dataset.UCF101_FLOW(path[data + '-class_idx'], split=path[data + '-test_split'],
                                                 flow_root=path[data + '_flows'], train=False, flip=False,
                                                 clip_len=clip_len)
    top_k = conf['top_k']

    clients = []
    server = FL_server.server(conf, test_dataset_rgb, test_dataset_flow, device)
    # bd_test = ba_evaluator(conf, server, path, device)

    attacker_num = conf['num_malicious_clients']
    for a in range(attacker_num):
        clients.append(Adversary(conf, train_dataset_rgb, train_dataset_flow, device, a))

    for c in range(attacker_num+1, conf["num_clients"]):
        clients.append(BenignClient(conf, train_dataset_rgb, train_dataset_flow, device, c))

    num_selected = conf["num_selected_clients"]
    e_lr = conf["lr"]
    decay_round = conf["decay_round"]
    eval_round = conf["eval_round"]

    for e in range(conf["global_epochs"]):
        # ********   attackers are selected at certain rounds    ********
        if (e+1) % 10 == 0:
            selected = random.sample(clients[attacker_num: conf["num_clients"]], num_selected-attacker_num)
            selected.extend(clients[0: attacker_num])
        else:
            selected = random.sample(clients, num_selected)

        # ********   randomly select any clients   ********
        # selected = random.sample(clients, num_selected)

        # Update global lr
        if e == decay_round:
            e_lr = e_lr / 10
            decay_round = decay_round*2
        if rgb:
            client_local_train(selected, server, e_lr, 'rgb', e)
        if flow:
            client_local_train(selected, server, e_lr, 'flow', e)
        if (e+1) % eval_round == 0:
            server.model_eval(e)
            # bd_test.ba_evaluate(e)

        if not os.path.exists('FL_Model'):
            os.makedirs('FL_Model')

        # Save global model every 50 rounds
        if (e+1) % 50 == 0:
            if rgb:
                torch.save(server.global_rgb.state_dict(), 'FL_Model/%s-%dx%d-ba_rgb_%d-Aggregations_lr-%.3f' % (t, conf['ts_x'], conf['ts_y'], e, e_lr) + '.pt')
            if flow:
                torch.save(server.global_flow.state_dict(), 'FL_Model/%s-%dx%d-ba_flow_%d-Aggregations_lr-%.3f' % (t, conf['ts_x'], conf['ts_y'], e, e_lr) + '.pt')

    server.model_eval(conf["global_epochs"])
    server.writer.close()
