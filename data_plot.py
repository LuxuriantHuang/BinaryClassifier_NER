import matplotlib.pyplot as plt
import numpy as np


def data_plot(infile):
    infopen = open(infile, 'r', encoding='utf-8')
    lines = infopen.readlines()
    list_i = []
    list_f1 = []
    for line in lines:
        data = line.split()
        list_i.append(int(data[0]))
        list_f1.append(float(data[1]) * 100)
    np_list_i = np.array(list_i)
    np_list_f1 = np.array(list_f1)
    # print(np_list_i)
    # print(np_list_f1)

    plt.legend(loc='upper right')
    plt.title('F1-Measure in Verification Set')
    plt.xlabel('i')
    plt.ylabel('F1-measure/%')
    plt.plot(np_list_i, np_list_f1, label='f1-measure')
    plt.show()

    infopen.close()


if __name__ == '__main__':
    data_plot("./sets/verification_out.txt")
