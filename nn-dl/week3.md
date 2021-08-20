# week 3 

input layer -> hidden layers -> output layers 

Hiden Layer: is not observed 


Essentially, each node in a hidden layer computes a logistic regression with an output a. Each subsequent layer then computes a logistic regression from those predictions. 

Question: What makes a[1],1 differ from a[1],2 where different nodes in the same layer have the same inputs but a different output

### Different Activation Functions 

sigmoid is a linear function between 0 and 1 

hyperbolic tangent function: a = tanh(z), goes between -1, 1. 

tanh(z) = (exp(z) - exp(-z))/(exp(z) + exp(-z)). The means are closer to 0 and centers the data. Makes learning easier in the next layer because data is centered. 

Still used sigmoid function for the output layer for binary classification. Can you mix activation functions between hidden and output layers? 

For extreme values (large or small z) of z, slope of the tanh function is flat which slows gradient descent learning. 

ReLu: a=max(0,z). Fo negative values of Z are 0. The NN will learn much faster. 

Sigmoid: never use except for output layer of binary classfication. 

Relu and leaky Relu: Leahy relu has a bend where z < 0 = max(.01*z,z)

### why do we need a non linear activation function 

Then it just computes a linear function of W(1)x + b(1) and nullifies the purpose of the hidden layers. It defaults to a logistic regression

If a regression problem output between (-inf, inf), then a linear activation function in the output layer is needed. But OK to use relu (0,inf) for say housing prices or customer spend 

### slope of activation functions to find derivatives of backpropagation 

tanh: d/dz = 1- tanh(z)^2

relu: g(z) = max(0,z), d/dz = 0 if z <0,else 1


### gradient descent for neural networks 


### random initialization 

why: if you set to 0, then the hidden layer and output layer would be equal and no learning will occur. Called the symmetry problem. 

The hidden units are computing the same thing: this answers my question above. randomization makes a[1],1 != a[1],2 

if w starts out large, then this slows down learning because it starts out at high values of z. start out with small weight initialization between -3 and 3. 









