import torch
from torch.nn import functional as f

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def logistic_model(theta, x):
    a = torch.matmul(theta, x)
    return torch.sigmoid(a)


def get_loss(y_p, y_t):
    return -torch.mean(y_t * torch.log(y_p) + (1 - y_t) * torch.log(1 - y_p))


# def getx(triplets, i):
#     x1_pre = torch.tensor([triplets[i][0]], device=device)
#     x1 = f.one_hot(x1_pre, 501)
#     x2_pre = torch.tensor([triplets[i][1]], device=device)
#     x2 = f.one_hot(x2_pre, 501)
#     x3_pre = torch.tensor([triplets[i][2]], device=device)
#     x3 = f.one_hot(x3_pre, 501)
#     return torch.cat([x1, x2, x3], 1).float().t()


def getlist(infopen):
    lines = infopen.readlines()
    triplets = []
    ys = []
    for line in lines:
        nums = line.split()
        triplets.append([int(nums[0]), int(nums[1]), int(nums[2])])
        ys.append(int(nums[-1]))
    return triplets, ys


def OneHotProcessing(infile1, infile2, infile3, outfile):
    theta = torch.zeros(1, 501 * 3, device=device)
    infopen1 = open(infile1, 'r', encoding='utf-8')
    infopen2 = open(infile2, 'r', encoding='utf-8')
    infopen3 = open(infile3, 'r', encoding='utf-8')
    outfopen = open(outfile, 'w', encoding='utf-8')
    # lines = infopen1.readlines()
    # triplets_train = []
    # ys_train = []
    # for line in lines:
    #     nums = line.split()
    #     triplets_train.append([int(nums[0]), int(nums[1]), int(nums[2])])
    #     ys_train.append(int(nums[-1]))
    # lines = infopen2.readlines()
    # triplets_verification = []
    # ys_verification = []
    # for line in lines:
    #     nums = line.split()
    #     triplets_verification.append([int(nums[0]), int(nums[1]), int(nums[2])])
    #     ys_verification.append(int(nums[-1]))

    triplets_train, ys_train = getlist(infopen1)
    triplets_verification, ys_verification = getlist(infopen2)
    triplets_test, ys_test = getlist(infopen3)
    print("read in completed")

    alpha = 0.02
    batch_size = 5000
    f1_list = []
    for e in range(100):
        for i in range(0, len(triplets_train), batch_size):
            x = torch.tensor(triplets_train[i:i + batch_size], device=device)
            length = len(triplets_train[i:i + batch_size])
            # print(x.size())
            x = f.one_hot(x, 501).float().reshape(length, 501 * 3).t()
            # print('x.size()=', x.size())
            # print('theta.size()=', theta.size())
            y_p = logistic_model(theta, x)
            # print(y_p)
            # print('y_p.size()=', y_p.size())
            y = torch.tensor(ys_train[i:i + batch_size], device=device)
            # print(y)
            # print('y.size()=', y.size())
            loss = torch.matmul((y_p - y), x.t())
            # print(loss)
            theta.data = theta.data - alpha * loss.data
            # print(theta.size())
        # print(theta)

        # 验证集验证模型可信度
        true_predictions = 0  # 预测1正确
        one_predictions = 0  # 预测1的个数
        true_original = 0  # 原有1的个数

        # i = random.randint(0, len(triplets_train) - 1)
        x = torch.tensor(triplets_verification, device=device)
        length = len(triplets_verification)
        x = f.one_hot(x, 501).float().reshape(length, 501 * 3).t()
        y_p = logistic_model(theta, x).cpu().numpy().tolist()
        # print(y_p)
        # print(y.size())
        for i in range(len(triplets_verification)):
            if (y_p[0][i] >= 0.5) and (ys_verification[i] == 1):
                true_predictions += 1
            if y_p[0][i] >= 0.5:
                one_predictions += 1
            if ys_verification[i] == 1:
                true_original += 1

        recall_rate = float(true_predictions) / float(true_original)
        precision_rate = float(true_predictions) / float(one_predictions)
        if recall_rate != 0 or precision_rate != 0:
            f1_measure = 2 * (recall_rate * precision_rate) / (recall_rate + precision_rate)
        else:
            f1_measure = 0.0

        f1_list.append([e, f1_measure])

        if e % 10 == 0:
            print(str(e) + ':')
            print("recall rate:", recall_rate)
            print("precision_rate:", precision_rate)
            print("f1 measure:", f1_measure)
    for list in f1_list:
        outfopen.write(str(list[0]+1) + ' ' + str(list[1]) + '\n')

    # 测试集测试模型训练效果
    true_predictions = 0  # 预测1正确
    one_predictions = 0  # 预测1的个数
    true_original = 0  # 原有1的个数

    # i = random.randint(0, len(triplets_train) - 1)
    x = torch.tensor(triplets_test, device=device)
    length = len(triplets_test)
    x = f.one_hot(x, 501).float().reshape(length, 501 * 3).t()
    y_p = logistic_model(theta, x).cpu().numpy().tolist()
    # print(y_p)
    # print(y.size())
    for i in range(len(triplets_test)):
        if (y_p[0][i] >= 0.5) and (ys_test[i] == 1):
            true_predictions += 1
        if y_p[0][i] >= 0.5:
            one_predictions += 1
        if ys_test[i] == 1:
            true_original += 1

    print("In test set:")
    recall_rate = float(true_predictions) / float(true_original)
    print("recall rate:", recall_rate)
    precision_rate = float(true_predictions) / float(one_predictions)
    print("precision_rate:", precision_rate)
    if recall_rate != 0 or precision_rate != 0:
        f1_measure = 2 * (recall_rate * precision_rate) / (recall_rate + precision_rate)
    else:
        f1_measure = 0.0
    print("f1 measure:", f1_measure)

    infopen1.close()
    infopen2.close()
    infopen3.close()
    outfopen.close()


if __name__ == '__main__':
    OneHotProcessing("./sets/pre_onehot_train_set.txt",
                     "./sets/pre_onehot_verification_set.txt",
                     "./sets/pre_onehot_test_set.txt",
                     "./sets/verification_out1.txt")
