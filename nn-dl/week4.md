# Deep L-layer Neural Network 

Notation of a 4 layer neural network:
L = # of layers in the network (hidden layers plus output )
n<super>[l]</super> = # of units in layer l (units = nodes)

a[l] = activations in layer l 

a[l] = g[l]z[l]

z[l] = W[l]*a[l-1] + b[l]

a[l] = g[l]{z[l]}

```python
Z1 = W1 @ X + b1
A1 = np.tanh(Z1)

explicit for loop from l..L for forward propagation
```

## getting your matrix dimensions right

Dimensions of 

Z[l] = (n[l],1);

W[l] = (n[l], n[l-1]), 
 
X = (n[0], 1)

b[l] = (n[l], 1)

Derivatives are the W and b equal their forward propagation matrices

Dimensions of a vectorized solution: 

Z[1] = (n[1], m); m = # of observations 
A[1] = (n[1], m)
W[1] = same as above 
X = (n[0], m)
b[1] = (n[1], m)

### why deep network 

Simple to complex representation as the layers progess. The first layer might find edges of a picture, whereas the later layers will form those edges to eyes, nose, etc. 

Circuit Theory: There are functions you can compute with a small L layer deep neural network that a shallower networks require exponentially more hidden units to compute 

Number of hidden units = log(n) for deep NN, or 2**n for one hidden layer

### Building Blocks of NN 

Forward and backward functions compute a[0] -> a[1], then cache Z[1] to compute da[l] -> da[l-1]

### Forward and Backward propagation 

backward propagation computes derivatives 

### hyperparameters and parameters 
 parameters: W and b 

 hyperparameters: learning rate, iterations, hidden layers, hidden units, choice of acitvation function, momentum term, regulations, etc. 

 



