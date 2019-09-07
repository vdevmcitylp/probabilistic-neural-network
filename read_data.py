from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale

def input():

	iris = datasets.load_iris()
	x = iris.data
	y = iris.target

	x = scale(x)

	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 0)

	data = {'x_train': x_train, 
			'x_test': x_test, 
			'y_train': y_train, 
			'y_test': y_test}

	return data
