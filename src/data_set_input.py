from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np
import pandas as pd
import os
import numpy as np
import csv
import glob
import pandas as pd
import os
import os


def open_file(path,encode):
    csv = pd.read_csv(path, encoding=encode)
    csv = csv.values
    return csv


def data_set_class(data):
    Accident_set = []
    Non_Accident_set = []
    for i in data:
        if (i[1] == 1):
            name = i[0]
            # print(name)
            filename = name.strip('.mp4')
            file_data = open_file('task1/%s/%s-mp4-acc.csv' % (filename, filename), 'utf-8')
            for j in file_data:
                time = str(j[0])
                x = np.float32(j[1])+10
                y = np.float32(j[2])+10
                z = np.float32(j[3])+10
                Accident_set.append([x,y,z])

        elif (i[1] == 0):
            name = i[0]
            # print(name)
            filename = name.strip('.mp4')
            file_data = open_file('task1/%s/%s-mp4-acc.csv' % (filename, filename), 'utf-8')
            for j in file_data:
                for k in j:
                    time = str(j[0])
                    x = np.float32(j[1])+10
                    y = np.float32(j[2])+10
                    z = np.float32(j[3])+10
                    Non_Accident_set.append([x, y, z])



    return Accident_set,Non_Accident_set
    # print(Accident_set)

def label_set(set,label):

    N = len(set)
    print(N)
    ano_data = []

    for i in range(N):
        ano_data.append(label)

    yy = np.array(ano_data)
    return yy

def save_file(acc_set,non_acc_set):

    for i, label in enumerate(["acc", "non_acc"]):
        # 3가지에 이것에 대해 하겠다.(3개의 문자열 원소를 가진 리스트)
        # data = enumerate([1, 2, 3])
        # for i, value in data:
        #	print(i, ":", value)
        # print()

        # 파일 패스 csv 파일 읽어오고 3가지에 대해서 (먼저 웤부터)

        # 어노테이션 csv 파일을 하나 더 생성함.

        # 아웃풋 파일이름: xx_ +윈도우 사이즈_ +threshold_ + label .csv
        outputfilename1 = "./input_files/xx_"  + label + ".csv"
        outputfilename2 = "./input_files/yy_"  + label + ".csv"

        # x,y = dataimport (파일 패스1,2 에서) 임포트 한다. 다시 이게 진짜인듯
        # x,y = dataimport(filename,path1)  # 여기서 x와 y를 만든다.
        if(label == 'acc'):

            with open(outputfilename1, "w") as f:
                writer = csv.writer(f, lineterminator="\n")  # 쓰고 다음줄로 넘어가나보다.
                writer.writerows(acc_set)  # x를 쓴다.
            with open(outputfilename2, "w") as f:
                writer = csv.writer(f, lineterminator="\n")
                labeling = [1,0,0]
                labeling = np.float32(labeling)
                y = label_set(acc_set,labeling)
                writer.writerows(y)  # y를 쓴다.
            print(label + "finish!")
        elif label == "non_acc":

            with open(outputfilename1, "w") as f:
                writer = csv.writer(f, lineterminator="\n")  # 쓰고 다음줄로 넘어가나보다.
                writer.writerows(non_acc_set)  # x를 쓴다.
            with open(outputfilename2, "w") as f:
                writer = csv.writer(f, lineterminator="\n")
                labeling = [0,1,0]
                labeling = np.float32(labeling)
                y = label_set(non_acc_set, labeling)
                writer.writerows(y)  # y를 쓴다.
            print(label + "finish!")

# main
if __name__ == '__main__':
    data_set_label = open_file('task1/data_set_01_labeling_result.csv', 'EUC_KR')
    accident_set, Non_accident_set = data_set_class(data_set_label)
    # print(accident_set)
    # accident distance
    acc_dis = []
    # Non_accident distance
    Nonacc_distance = []

    # draw_3D(accident_set,Non_accident_set)

    save_file(accident_set,Non_accident_set)

    fo = open_file('input_files/xx_acc.csv', 'utf-8')

    for i in fo:
        print(type(i[0]))



