from sklearn.preprocessing import OneHotEncoder
def one_hot_set(infile1, infile2, outfile):
    infopen1 = open(infile1, 'r', encoding='utf-8')
    infopen2 = open(infile2, 'r', encoding='utf-8')
    outfopen = open(outfile, 'w', encoding='utf-8')
    lines = infopen1.readlines()
    onehotArr = []
    for line in lines:
        line = line.strip()
        onehotArr.append(line)
    lines = infopen2.readlines()
    trainarr_pre = []
    for line in lines:
        head, sep, tail = line.partition('/')
        try:
            loc = onehotArr.index(head)
        except ValueError:
            trainarr_pre.append(5000)
        else:
            trainarr_pre.append(loc)
    trainarr = []
    for i in range(len(trainarr_pre)):
        #     outfopen.write(str(trainarr_pre[i])+'\n')
        if i == 0:
            trainarr.append([5000, trainarr_pre[i], trainarr_pre[i + 1]])
        elif i == len(trainarr_pre) - 1:
            trainarr.append([trainarr_pre[i - 1], trainarr_pre[i], 5000])
        else:
            trainarr.append([trainarr_pre[i - 1], trainarr_pre[i], trainarr_pre[i + 1]])
    # for i in range(len(trainarr_pre)):
    #     outfopen.write(str(trainarr[i]) + '\n')

    infopen1.close()
    infopen2.close()
    outfopen.close()


# one_hot_set("1train_frequency3.txt", "1train1.txt", "1train_onehot0.txt")
