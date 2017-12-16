import numpy as np
import read_data, performance_metrics

# np.random.seed(45)

def rbf(centre, x, sigma):
	temp = -np.sum((centre - x) ** 2, axis = 1)
	temp = temp / (2 * sigma * sigma)
	temp = np.exp(temp)
	return np.sum(temp)

def PNN(X_train, Y_train, X_test, Y_test, num_class):

	num_testset = X_test.shape[0]
	X_train_class = []

	for i in range(num_class):
		index = np.where(Y_train == i)
		# print(index)
		X_train_class.append(X_train[index, :])
		# print(X_train_class[i])		
	
	g = np.zeros(num_class)
	pred = np.zeros(num_testset)

	for i in range(num_testset):
		# if(i%1000 == 0):
		# 	print(i),
		for j in range(num_class):
			g[j] = np.sum(rbf(X_test[i].reshape(1, -1), X_train_class[j][0], 1.5)) / X_train_class[j][0].shape[0] 
			# print(X_train_class[j][0].shape)
		pred[i] = np.argmax(g)
		# print(pred[i])

	return pred

def output(x_in_out_train_split, y_in_out_train_split, x_test, y_test, num_PNN, num_class):

	result = []
	for i in range(num_PNN):
	 	result.append(PNN(x_in_out_train_split[i], y_in_out_train_split[i], x_test, y_test, num_class))

	vote = np.zeros(y_test.size)
	for i in range(num_PNN):
		vote = vote + result[i]

	label = [int(v) for v in vote > num_PNN/2]
	return label
	# print(result)
	# with open('result_HTRU', 'wb') as f:
	# 	pickle.dump(result, f)