def pretreatmentByWordFrequency(infile, outfile):
    infopen = open(infile, 'r', encoding='utf-8')
    outfopen = open(outfile, 'w', encoding='utf-8')
    inlines = infopen.readlines()
    dict = {}
    for line in inlines:
        for word in line.split():
            word = word.replace('[', '').replace(']', '')
            a = word.split('/')
            if a[0] not in dict:
                dict[a[0]] = 1
            else:
                dict[a[0]] += 1
    k = 0
    for item in sorted(dict, key=dict.__getitem__, reverse=True):
        # if 'n' in dict2[item] and dict2[item].__len__() <= 2:
        k += 1
        if k <= 500:
            # head, sep, tail = item.partition('{')
            # if head[-1] != '\n':
            #     outfileopen.write(head + '\n')
            # else:
            #     outfileopen.write(head)
            if item[-1] != '\n':
                outfopen.write(item + '\n')
            else:
                outfopen.write(item)
    outfopen.write("Others\n")
    infopen.close()
    outfopen.close()


def main():
    pretreatmentByWordFrequency("./sets/train_set.txt", "./sets/1train_word_frequency_sequence.txt")


if __name__ == '__main__':
    main()
