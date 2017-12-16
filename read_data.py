import numpy as np
from sklearn.datasets import fetch_mldata
import pandas as pd

# def input():
# 	f = open('../../datasets/Corners.txt', 'r')

# 	info = f.readlines()

# 	X = []
# 	Y = []

# 	for item in info:

# 		item = item.split()
# 		x1 = float(item[0])
# 		x2 = float(item[1])
# 		y = float(item[2])

# 		x = np.append(x1, x2)

# 		X.append(x)
# 		Y.append(y)

# 	X = np.array(X)
# 	Y = np.array(Y)

# 	return X, Y
		
# def input():
# 	mnist = fetch_mldata('MNIST original')
# 	X = mnist.data
# 	Y = mnist.target.reshape(X.shape[0], 1)

# 	index = np.random.choice(70000, 10000, replace = False)

# 	X = X[index, :]
# 	Y = Y[index, :]
# 	return X, Y

def input():
	
	data = pd.read_excel('../../datasets/One/creditcard.xlsx', header = None)

	y = data.iloc[:, -1]
	x = data.iloc[:, :-1]

	x = np.array(x)
	y = np.array(y)

	return x, y