import numpy as np

def confusion_matrix(target, predicted, num_class):
	
	# target_nb = np.argmax(target, axis = 1)
	# predicted_nb = np.argmax(predicted, axis = 1)

	target_nb = target
	predicted_nb = predicted
	
	num_trainset = target_nb.size

	cm = np.zeros((num_class, num_class))
	for i in range(num_trainset):
		cm[target_nb[i], predicted_nb[i]] += 1
	
	return cm

def accuracy(target, predicted):

	temp = target - predicted
	non_zero = np.count_nonzero(temp)
	zero = target.shape[0] - non_zero

	return float(zero) / target.shape[0]

def precision(target, predicted, num_class):
	
	num_trainset = target.shape[0]
	cm = confusion_matrix(target, predicted, num_class)

	p = np.zeros((num_class, 1))

	for i in range(num_class):
		p[i] = cm[i, i]/np.sum(cm[:, i])

	return np.mean(p)

def recall(target, predicted, num_class):
	num_trainset = target.shape[0]
	cm = confusion_matrix(target, predicted, num_class)

	r = np.zeros((num_class, 1))

	for i in range(num_class):
		r[i] = cm[i, i]/np.sum(cm[i, :])

	return np.mean(r)

# def loss(Y, output_layer, num_trainset):
# 	return np.sum((Y - output_layer) ** 2) / num_trainset	

# def sum_error(error, num_layers):
# 	sum_square_error = np.zeros((num_layers, 1)) # 0 not to be used
# 	for i in xrange(num_layers):
# 		sum_square_error[i] = (np.sum(error[i]**2))**0.5

# 	return sum_square_error