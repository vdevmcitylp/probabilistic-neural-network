# probabilistic-neural-network

A probabilistic neural network (PNN) is a feedforward neural network, which is widely used in classification and pattern recognition problems.

Refer [here](https://en.wikipedia.org/wiki/Probabilistic_neural_network) and [here](http://www.personal.reading.ac.uk/~sis01xh/teaching/CY2D2/Pattern3.pdf) for the theory. 

PNN's are lazy learners, there's no training step involved. We perform a forward pass when we want to classify a test point.

A PNN has four layers:
1. Input Layer
2. Pattern (Hidden) Layer
3. Summation Layer
4. Output Layer

![PNN Diagram](https://raw.githubusercontent.com/vdevmcitylp/probabilistic-neural-network/master/pnn.JPG "PNN Diagram")

The code for this tutorial is available [here](https://github.com/vdevmcitylp/probabilistic-neural-network/blob/master/pnn.py).

### Input Layer

The input layer is the feature vector representation of the input. Normalize the data before feeding it to the network.
So far so good.

### Pattern (Hidden) Layer

The number of nodes in this layer is equal to the number of *training points* in your dataset. For instance, if your training set has 100,000 points, then we'll have 100,000 nodes in this layer.

### Summation Layer

The number of nodes in this layer is equal to the number of *classes*. So for two-class classfication, we'll have two nodes.

To compute the activation for a particular node (class) in this layer, we apply this formula.

Lines 5-10 implement this formula,

    def rbf(centre, x, sigma):
    
      temp = -np.sum((centre - x) ** 2, axis = 1)
      temp = temp / (2 * sigma * sigma)
      temp = np.exp(temp)
      return np.sum(temp)

'centre' is the test point which is to be classified.

We call the 'rbf' function for each class and get the activations for the summation layer. 

Lines 33-35 do this,

    for j in range(num_class):
			  # Calculate summation layer
			  g[j] = np.sum(rbf(X_test[i].reshape(1, -1), X_train_class[j][0], 1.5)) / X_train_class[j][0].shape[0]
      
### Output Layer

This layer has one node. Nothing happens in this layer really, we just predict as output the class for which the summation layer has the maximum activation value.

Line 38

    pred[i] = np.argmax(g)
    
The rest of the code is for checking the performance of the algorithm, the metrics are implemented in [this](https://github.com/vdevmcitylp/probabilistic-neural-network/blob/master/performance_metrics.py) file.

And it's as simple as that! Feel free to open an issue if required.
