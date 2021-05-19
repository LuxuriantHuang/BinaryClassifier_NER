import torch
from torch.nn import functional as f
import random

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
theta = torch.randn(1, 1503, device=device, requires_grad=True)


def logestic_model(x):
    a = torch.mm(theta, x)
    return torch.sigmoid(a)


def get_loss(y_p, y_t):
    return -torch.mean(y_t * torch.log(y_p) + (1 - y_t) * torch.log(1 - y_p))


def getx(triplets, i):
    x1_pre = torch.tensor([triplets[i][0]], device=device)
    x1 = f.one_hot(x1_pre, 501)
    x2_pre = torch.tensor([triplets[i][1]], device=device)
    x2 = f.one_hot(x2_pre, 501)
    x3_pre = torch.tensor([triplets[i][2]], device=device)
    x3 = f.one_hot(x3_pre, 501)
    return torch.cat([x1, x2, x3], 1).t().float()


def OneHotProcessing(infile1, infile2, outfile):
    infopen1 = open(infile1, 'r', encoding='utf-8')
    infopen2 = open(infile2, 'r', encoding='utf-8')
    outfopen = open(outfile, 'w', encoding='utf-8')
    lines = infopen1.readlines()
    triplets_train = []
    ys_train = []
    for line in lines:
        nums = line.split()
        triplets_train.append([int(nums[0]), int(nums[1]), int(nums[2])])
        ys_train.append(int(nums[-1]))

    lines = infopen2.readlines()
    triplets_verification = []
    ys_verification = []
    for line in lines:
        nums = line.split()
        triplets_verification.append([int(nums[0]), int(nums[1]), int(nums[2])])
        ys_verification.append(int(nums[-1]))
    print("read in completed")

    alpha = 0.02
    i = random.randint(0, len(triplets_train) - 1)
    # sum = torch.zeros(1, 1503, device=device)
    # for i in range(len(triplets_train)):
    x = getx(triplets_train, i)
    y_p = logestic_model(x)
    y = torch.tensor([ys_train[i]], device=device)
    # sum = sum + torch.mm(y_p - y, x.t())
    # theta.data = theta.data - alpha * sum.data
    loss = get_loss(y_p, y)
    # print(loss)
    loss.backward()
    # print(theta.grad)
    theta.data = theta.data - alpha * theta.grad.data
    print("go in training")

    # epoch
    # for e in range(len(triplets_train)):
    for e in range(100000):
        #     for i in range(len(triplets_train)):
        #         x = getx(triplets_train, i)
        #         y_p = logestic_model(x)
        #         y = torch.tensor([ys_train[i]], device=device)
        #         sum = sum + torch.mm(y_p - y, x.t())
        # theta.data = theta.data - alpha * sum.data
        i = random.randint(0, len(triplets_train) - 1)
        x = getx(triplets_train, i)
        y_p = logestic_model(x)
        y = torch.tensor([ys_train[i]], device=device)
        theta.grad.zero_()
        loss = get_loss(y_p, y)
        theta.data = theta.data - alpha * theta.grad.data

    # print('epoch: {}, loss: {}'.format(e * 500, loss.data.item()))

    for data in theta.cpu().detach().numpy().tolist():
        out = str(data) + ' '
        outfopen.write(out)
    outfopen.write('\n')

    # print(theta)
    print("learning completed")

    true_pridictions = 0  # 预测1正确
    one_predictions = 0  # 预测1的个数
    true_origional = 0  # 原有1的个数

    for i in range(len(triplets_verification)):
        x = getx(triplets_verification, i)
        y_p = logestic_model(x)
        y = torch.tensor([ys_verification[i]], device=device)

        loss = get_loss(y_p, y)

        out = str(i) + ' y_p:' + str(y_p.item()) + ' y:' + str(y.item()) + ' T/F:'
        if ((y_p.item() >= 0.5) and (y.item() == 1)) or ((y_p.item() < 0.5) and (y.item() == 0)):
            out += 'T\n'
        else:
            out += 'F\n'
        outfopen.write(out)

        if (y_p.item() >= 0.5) & (y.item() == 1):
            true_pridictions += 1
        if (y_p.item() >= 0.5):
            one_predictions += 1
        if (y.item() == 1):
            true_origional += 1

    print("total:", len(triplets_verification))
    print("true pridictions:", true_pridictions)
    print("one pridictions:", one_predictions)
    print("true oritional:", true_origional)
    recall_rate = float(true_pridictions) / float(true_origional)
    precision_rate = float(true_pridictions) / float(one_predictions)
    f1_measure = 2 * (recall_rate * precision_rate) / (recall_rate + precision_rate)
    print("recall rate:", recall_rate)
    print("precision_rate", precision_rate)
    print("f1 measure:", f1_measure)


if __name__ == '__main__':
    OneHotProcessing("./sets/pre_onehot_train_set.txt",
                     "./sets/pre_onehot_verification_set.txt",
                     "./sets/verification_out.txt")
