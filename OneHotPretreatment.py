def OneHotPretreatment(infile1, infile2, outfile):
    infopen1 = open(infile1, 'r', encoding='utf-8')
    infopen2 = open(infile2, 'r', encoding='utf-8')
    outfopen = open(outfile, 'w', encoding='utf-8')
    freq_list = []
    dich_list_word = []
    dich_list_yes = []
    lines = infopen1.readlines()
    for line in lines:
        freq_list.append(line.replace('\n', ''))
    lines = infopen2.readlines()
    for line in lines:
        word = line.split()
        if word[0] in freq_list:
            index = freq_list.index(word[0])
            dich_list_word.append(index)
        else:
            dich_list_word.append(500)
        dich_list_yes.append(int(word[-1]))
    # print(freq_list[78])
    # print(dich_list_word[174], dich_list_yes[174])

    # dich_list_word_onehot=[]
    # for i in range(len(dich_list_word)):
    #     label=torch.tensor([dich_list_word[i]])
    #     num_class=5001
    #     label_onehot=functional.one_hot(label,num_classes=num_class)
    #     dich_list_word_onehot.append(label_onehot)
    # print(dich_list_word_onehot[174])

    Triplets_words = []
    Triplets_yes = dich_list_yes
    for i in range(len(dich_list_word)):
        if i == 0:
            Triplets_words.append([500, dich_list_word[i], dich_list_word[i + 1]])
        elif i == len(dich_list_word) - 1:
            Triplets_words.append([dich_list_word[i - 1], dich_list_word[i], 500])
        else:
            Triplets_words.append([dich_list_word[i - 1], dich_list_word[i], dich_list_word[i + 1]])
    # print(Triplets_words[174],Triplets_yes[174])

    for i in range(len(Triplets_words)):
        outfopen.write(str(Triplets_words[i][0]) + ' '
                       + str(Triplets_words[i][1]) + ' '
                       + str(Triplets_words[i][2]) + ' '
                       + str(Triplets_yes[i]) + '\n')
    infopen1.close()
    infopen2.close()
    outfopen.close()


def main():
    OneHotPretreatment("./sets/1train_word_frequency_sequence.txt",
                       "./sets/1train_nt_dichotomy.txt",
                       "./sets/pre_onehot_train_set.txt")
    OneHotPretreatment("./sets/1train_word_frequency_sequence.txt",
                       "./sets/2verification_nt_dichotomy.txt",
                       "./sets/pre_onehot_verification_set.txt")
    OneHotPretreatment("./sets/1train_word_frequency_sequence.txt",
                       "./sets/3test_nt_dichotomy.txt",
                       "./sets/pre_onehot_test_set.txt")


if __name__ == '__main__':
    main()
