import numpy as np
import csv
import glob
import os

window_size = 1000
threshold = 60
slide_size = 200 #less than window_size!!!

def dataimport(path1, path2):
	xx = np.empty([0,window_size,90],float)
	yy = np.empty([0, 4], float) #### given matrix is a number of class
	#초기화 xx는 3차원 ,yy는 2차원 ,맨앞이 0차원으로써 메모리만 만들어주는듯.

	###Input data###
	#data import from csv
	input_csv_files = sorted(glob.glob(path1))

	for f in input_csv_files:
		print("input_file_name=",f)

		#v를 읽는데, v 안에 elm
		data = [[ float(elm) for elm in v] for v in csv.reader(open(f, "r"))]
		#2차원 리스트에 CSV 원소 하나씩 넣기
		tmp1 = np.array(data)
		#NP.array형식으로 변환 tmp1

		x2 = np.empty([0,window_size,90],float)
		#x2는 empty로 메모리 잡아주기

		#data import by slide window
		k = 0
		while k <= (len(tmp1) + 1 - 2 * window_size):
			x = np.dstack(np.array(tmp1[k:k+window_size, 0:90]).T)
			''' 데이터변환
			a = np.array((1,2,3))
			b = np.array((2,3,4))
			np.dstack((a,b))
			array([[[1, 2],
					[2, 3],
					[3, 4]]])
			a = np.array([[1],[2],[3]])
			b = np.array([[2],[3],[4]])
			np.dstack((a,b))
			array([[[1, 2]],
				   [[2, 3]],
				   [[3, 4]]])
			'''
			x2 = np.concatenate((x2, x),axis=0)
			#이어 붙이기
			'''
			a = np.array([[1, 2], [3, 4]])
			b = np.array([[5, 6]])
			np.concatenate((a, b), axis=0)
			array([[1, 2],
				   [3, 4],
				   [5, 6]])	
			'''
			k += slide_size
		#xx에 x2 이어붙이기
		xx = np.concatenate((xx,x2),axis=0)
	xx = xx.reshape(len(xx),-1)
	#xx의 길이에 따라 행 맞춰주기 reshape
	'''
	x = np.arange(12)
	x = x.reshape(3,4)
	x
	array([[ 0,  1,  2,  3],
		   [ 4,  5,  6,  7],
		   [ 8,  9, 10, 11]])
		   
	x.reshape(4,-1)
array([[ 0,  1,  2],
       [ 3,  4,  5],
       [ 6,  7,  8],
       [ 9, 10, 11]])
       x.reshape(3,-1)
array([[ 0,  1,  2,  3],
       [ 4,  5,  6,  7],
       [ 8,  9, 10, 11]])
       
       x.reshape(-1,3)
array([[ 0,  1,  2],
       [ 3,  4,  5],
       [ 6,  7,  8],
       [ 9, 10, 11]])
       
       x.reshape(-1,2)
array([[ 0,  1],
       [ 2,  3],
       [ 4,  5],
       [ 6,  7],
       [ 8,  9],
       [10, 11]])
	'''

	###Annotation data###
	#data import from csv
	annotation_csv_files = sorted(glob.glob(path2))
	for ff in annotation_csv_files:
		print("annotation_file_name=",ff)
		ano_data = [[ str(elm) for elm in v] for v in csv.reader(open(ff,"r"))]
		tmp2 = np.array(ano_data)


		#아니 근데 이거 왜함? 윈도우사이즈에 따라서 왜함?? y하는데 ..anodata 있어서 그런가

		#data import by slide window
		y = np.zeros(((len(tmp2) + 1 - 2 * window_size)//slide_size+1, 4)) #### the last parameter should be the number of class
		k = 0
		while k <= (len(tmp2) + 1 - 2 * window_size):
			y_pre = np.stack(np.array(tmp2[k:k+window_size]))
			walking =0 #modified
			laydown = 0
			sit = 0
			for j in range(window_size):
				if y_pre[j] == "walk": #modified
					walking += 1
				elif y_pre[j] == "laydown": #modified
					laydown += 1
				elif y_pre[j] == "sit": #modified
					sit += 1

			#딱 보니까 원핫 인코더로 하는듯
			if walking > window_size * threshold / 100: #modified
				y[k // slide_size, :] = np.array([0, 1, 0, 0])
			elif laydown > window_size * threshold / 100: #modified
				y[k // slide_size, :] = np.array([0, 0, 1, 0])
			#엠티는 안했어 따로
			elif empty > window_size * threshold / 100: #modified
				y[k // slide_size, :] = np.array([0, 0, 0, 1])
			else:
				y[k//slide_size,:] = np.array([2,0,0,0]) # should not be deleted
			#2로 원핫된것은 잘못된것인듯
			k += slide_size

		yy = np.concatenate((yy, y),axis=0)

	#여기서 shape을 알려주네.
	print(xx.shape,yy.shape)
	return (xx, yy)


#### Main ####
if not os.path.exists("input_files/"):
	os.makedirs("input_files/")
	#폴더 만들기
for i, label in enumerate(["walk", "laydown", "sit"]):
	#3가지에 이것에 대해 하겠다.(3개의 문자열 원소를 가진 리스트)
	#data = enumerate([1, 2, 3])
	#for i, value in data:
	#	print(i, ":", value)
	#print()

	#파일 패스 csv 파일 읽어오고 3가지에 대해서 (먼저 웤부터)
	filepath1 = "./190528_Dataset2/190528_" + str(label) + "*.csv"
	#주석(정답)
	filepath2 = "./190528_Dataset2/annotation_" + str(label) + "*.csv"

	#아웃풋 파일이름: xx_ +윈도우 사이즈_ +threshold_ + label .csv
	outputfilename1 = "./input_files/xx_" + str(window_size) + "_" + str(threshold) + "_" + label + ".csv"
	outputfilename2 = "./input_files/yy_" + str(window_size) + "_" + str(threshold) + "_" + label + ".csv"

	#x,y = dataimport (파일 패스1,2 에서) 임포트 한다. 다시 이게 진짜인듯
	x, y = dataimport(filepath1, filepath2) #여기서 x와 y를 만든다.

	# with open("foo.txt", "w") as f:
	# 	f.write("Life is too short, you need python")
	# 위와
	# 같이
	# with문을
	# 사용하면
	# with 블록을 벗어나는 순간 열린 파일 객체 f가 자동으로 close되어 편리하다

	#파일연다. 쓰기모드로
	with open(outputfilename1, "w") as f:
		writer = csv.writer(f, lineterminator="\n") #쓰고 다음줄로 넘어가나보다.
		writer.writerows(x) #x를 쓴다.
	with open(outputfilename2, "w") as f:
		writer = csv.writer(f, lineterminator="\n")
		writer.writerows(y) #y를 쓴다.
	print(label + "finish!")
	#walk , laydown, sit 이 3가지에 대해 하나씩 끝낸다.