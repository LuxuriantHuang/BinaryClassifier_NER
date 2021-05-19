import torch
from torch.nn import functional as f
from math import exp
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

    alpha = 0.02

    i = random.randint(0, len(triplets_train) - 1)
    x = getx(triplets_train, i)

    y_p = logestic_model(x)
    y = torch.tensor([ys_train[i]], device=device)

    loss = get_loss(y_p, y)
    # print(loss)

    loss.backward()

    # print(theta.grad)

    theta.data = theta.data - alpha * theta.grad.data

    #epoch
    for e in range(200):
        for _ in range(500):
            i = random.randint(0, len(triplets_train) - 1)
            x = getx(triplets_train, i)

            y_p = logestic_model(x)
            y = torch.tensor([ys_train[i]], device=device)
            loss = get_loss(y_p, y)

            theta.grad.zero_()
            loss.backward()

            theta.data=theta.data-alpha*theta.grad.data
        print('epoch: {}, loss: {}'.format(e*500, loss.data.item()))

    print(theta)

    # outfopen.write(str(x.cpu().numpy().tolist()))
    # theta_x = torch.mm(theta, x)
    # y_pred = torch.sigmoid(theta_x)
    # y = torch.tensor([ys_train[i]], device=device)
    #
    # # loss = torch.mm((y_pred - y), (x.t()))
    # loss = y_pred - y
    #
    # theta = theta - alpha * (torch.mm(loss, x.t()))
    # if k >= 40000:
    #     outfopen.write(str(k) + ' '
    #                    + 'i:' + str(i) + ' '
    #                    + 'theta_x:' + str(theta_x.cpu().item()) + ' '
    #                    + 'y_pred:' + str(y_pred.item()) + ' '
    #                    + 'y:' + str(y.item()) + ' '
    #                    + 'loss:' + str(loss.item()) + '\n')

    # loss.backward()

    # with torch.no_grad():
    #     theta-=alpha*theta.grad()
    #
    #     theta.grad.zero_()

    infopen1.close()
    infopen2.close()
    outfopen.close()


def main():
    OneHotProcessing("./sets/pre_onehot_train_set.txt",
                     "./sets/pre_onehot_verification_set.txt",
                     "./sets/verification_out.txt")


if __name__ == '__main__':
    main()
