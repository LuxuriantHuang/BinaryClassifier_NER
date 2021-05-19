def pretreatmentByNtCharacteristic(infile, outfile):
    infopen = open(infile, 'r', encoding='utf-8')
    outfopen = open(outfile, 'w', encoding='utf-8')
    inlines = infopen.readlines()
    list = []
    flag = 0
    for line in inlines:
        for words in line.split():
            if '[' in words:
                flag = 1
                word = words.replace('[', '')
                head,sep,tail = word.partition('/')
                list.append([head, 1])
            else:
                if ']' in words:
                    # 这里实际上用了简化处理，经统计样本中存在[]的情况仅有nt类和i类两种
                    # 而由于实际上i类的情况十分少，因而采用了整体全部视为nt类的粗略处理方法
                    # word = words.split(']')
                    # if words[1].split() == "nt":
                    flag = 0
                    head,sep,tail = words.partition('/')
                    list.append([head, 1])
                    # else:
                    #     word = words[0].split('/')
                    #     list.append([word[0].split(), word[1].split()])
                else:
                    if flag == 1:
                        head, sep, tail = words.partition('/')
                        list.append([head, 1])
                    elif flag == 0:
                        head, sep, tail = words.partition('/')
                        if tail.split()=='nt':
                            list.append([head, 1])
                        else:
                            list.append([head, 0])
    for item in list:
        outfopen.write(str(item[0])+' '+str(item[1])+ '\n')
    infopen.close()
    outfopen.close()


def main():
    pretreatmentByNtCharacteristic("./sets/train_set.txt", "./sets/1train_nt_dichotomy.txt")
    pretreatmentByNtCharacteristic("./sets/verification_set.txt", "./sets/2verification_nt_dichotomy.txt")
    pretreatmentByNtCharacteristic("./sets/test_set.txt", "./sets/3test_nt_dichotomy.txt")


if __name__ == '__main__':
    main()
