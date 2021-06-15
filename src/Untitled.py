import numpy as np
import csv
import glob
import os
xx = np.empty([2,3,5],float)


input_csv_files = sorted(glob.glob("./123.csv"))

for f in input_csv_files:
    # for v in csv.reader(open(f, "r")):
    #     print(v)
    #print(data)

    #문자열로 읽어지는 v
    #이렇게하니 for 앞이 사라지는구나.
    #data = [float(v) in v] for v in csv.reader(open(f, "r")) #2번, 그리고 1번. 넣는다.
    #print(data)
    #
    # data = [[float(elm) for elm in v] for v in csv.reader(open(f,"r_"))]
    # #리스트로 감싸주면 몇번할건지 안사라진다.
    # data =[[float(elm) for elm in v] for v in csv.reader(open(f, "r"))]
    a = "123"
    a = ['123','40','2']

    for elm in a:
        for el in elm:
            print(el)
    print(a)