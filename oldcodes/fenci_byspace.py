def fenci_byspace(infile, outfile):
    infopen = open(infile, 'r', encoding='utf-8')
    outfopen = open(outfile, 'w', encoding='utf-8')
    lines = infopen.readlines()
    for line in lines:
        for db in line.split():
            if db not in outfile:
                outfopen.write(db + '\n')
    infopen.close()
    outfopen.close()
