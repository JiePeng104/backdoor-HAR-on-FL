import torch
import torch.utils.data
import torch.utils.tensorboard
import torch.nn.functional as F


def train(model, e_lr, loader, e, num_round, d, loss_e=100):
    optimiser = torch.optim.SGD(model.parameters(), lr=e_lr, momentum=0.9, weight_decay=1e-7)
    model.train(True)
    criterion = torch.nn.CrossEntropyLoss()
    for k in range(e):
        num_batch = 0
        running_loss = 0.0
        for batch_id, batch in enumerate(loader):
            num_batch += 1
            optimiser.zero_grad()
            data, target = batch
            data = data.to(d)
            target = target.to(d)
            output = model.forward(data)
            loss = criterion(output, target)
            loss.requires_grad_(True)
            loss.backward()
            optimiser.step()
            running_loss += loss.item()
            if num_batch % loss_e == 0:
                print("%s Round %d ,Epoch %d , %d batch Done!\n" % (model.mode, num_round, k, num_batch))
                print("%s Loss %f\n" % (model.mode, running_loss / 100))
                running_loss = 0.0
    return model


def top_k_contain(pre_tk, labels):
    pre_tk = torch.squeeze(pre_tk.data, 2)
    labels = torch.squeeze(labels.data, 1)
    correct = 0
    length = len(labels)
    for i in range(length):
        for t in pre_tk[i]:
            if labels[i].eq(t):
                correct += 1
                break
    return correct


def evaluate(model, loader, d, tk=5):
    loss = 0.0
    correct_t1 = 0
    correct_tk = 0
    num_batch = 0
    logits = []
    labels = []
    model.eval()
    # model.train(False)
    with torch.no_grad():
        for batch_id, batch in enumerate(loader):
            data, target = batch
            num_batch += 1
            data = data.to(d)
            target = target.to(d)
            output = model.forward(data)
            _, prediction = output.data.max(1)
            # print("t:")
            # print(target)
            correct_t1 += prediction.eq(target.data.view_as(prediction)).cpu().sum().item()
            _, pre_top_k = torch.topk(output.data, k=tk, dim=1)
            # print("p:")
            # print(pre_top_k)
            correct_tk += top_k_contain(pre_top_k, target)
            logits.append(output)
            labels.append(target)
            loss += F.cross_entropy(output, target).item()
        acc_t1 = 100 * float(correct_t1) / float(len(loader.dataset))
        acc_tk = 100 * float(correct_tk) / float(len(loader.dataset))
        loss = loss / float(num_batch)
    return acc_t1, acc_tk, loss, logits, labels


def two_stream_test(l1, l2, gt, sf_len, tk=5):
    correct_t1 = 0
    correct_tk = 0
    length = len(l1)
    for i in range(length):
        fusion = l1[i] + l2[i]
        sf = F.softmax(fusion.data, dim=1)
        _, prediction = sf.data.max(1)
        correct_t1 += prediction.eq(gt[i].data.view_as(prediction)).cpu().sum().item()
        _, pre_top_k = torch.topk(sf.data, k=tk, dim=1)
        correct_tk += top_k_contain(pre_top_k, gt[i])
    acc_t1 = 100 * float(correct_t1) / float(sf_len)
    acc_tk = 100 * float(correct_tk) / float(sf_len)
    return acc_t1, acc_tk
