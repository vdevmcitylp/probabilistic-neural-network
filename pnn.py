import numpy as np
import read_data, performance_metrics

# Helper function that essentially combines the hidden layer and summation layer
def rbf(centre, x, sigma):
	
	temp = -np.sum((centre - x) ** 2, axis = 1)
	temp = temp / (2 * sigma * sigma)
	temp = np.exp(temp)
	return np.sum(temp)

def PNN(X_train, Y_train, X_test, Y_test, num_class):

	num_testset = X_test.shape[0]
	X_train_class = []

	# Splits the training set into subsets where each subset contains data points from a particular class
	for i in range(num_class):
		index = np.where(Y_train == i)
		X_train_class.append(X_train[index, :])
	
	# Variable for storing the summation layer values from each class
	g = np.zeros(num_class)
	
	# Variable for storing the predictions for each test data point
	pred = np.zeros(num_testset)

	for i in range(num_testset):
		# Checking whether everything is running smoothly :P
		if(i%1000 == 0):
			print(i),
		
		for j in range(num_class):
			# Calculate summation layer
			g[j] = np.sum(rbf(X_test[i].reshape(1, -1), X_train_class[j][0], 1.5)) / X_train_class[j][0].shape[0] 
		
		# The index having the largest 'g' value is stored as the prediction
		pred[i] = np.argmax(g)
		
	return pred

# Write your own input function
# X, Y = read_data.input()

# Write your own split function
# X_train, Y_train, X_test, Y_test = split.split(X, Y)

# One Class Dataset
num_class = 2

#Call the PNN function for predictions
predictions = PNN(X_train, Y_train, X_test, Y_test, num_class)

#Performance Metrics
print(performance_metrics.accuracy(Y_test, predictions))
print(performance_metrics.confusion_matrix(Y_test, predictions, num_class))
print(performance_metrics.precision(Y_test, predictions, num_class))
print(performance_metrics.recall(Y_test, predictions, num_class))
