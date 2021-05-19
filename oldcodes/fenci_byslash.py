def fenci_byslash(infile, outfile):#去除词性
    infopen = open(infile, 'r', encoding='utf-8')
    outfopen = open(outfile, 'w', encoding='utf-8')
    lines = infopen.readlines()
    dict1 = {}
    dict2 = {}
    for line in lines:
        line = (line.replace('[', '')).replace(']','')
        a = line.split('/')
        if a[0] not in dict1:
            dict1[a[0]] = 1
        else:
            dict1[a[0]] += 1
        if a[0] not in dict2:
            dict2[a[0]] = a[1]
    k = 0
    for item in sorted(dict1, key=dict1.__getitem__, reverse=True):
        # if 'n' in dict2[item] and dict2[item].__len__() <= 2:
            k += 1
            if k <= 500:
                db = item
                outfopen.write(db + '\n')
    infopen.close()
    outfopen.close()

# fenci_byslash("1train1.txt","1train_frequency4.txt")