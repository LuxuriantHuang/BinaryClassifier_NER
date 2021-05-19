def fenci_byaa(infile, outfile):#去除{}
    infopen = open(infile, 'r', encoding='utf-8')
    outfopen = open(outfile, 'w', encoding='utf-8')
    lines = infopen.readlines()
    for line in lines:
        head, sep, tail = line.partition('{')
        if head[-1] != '\n':
            outfopen.write(head + '\n')
        else:
            outfopen.write(head)
    infopen.close()
    outfopen.close()

