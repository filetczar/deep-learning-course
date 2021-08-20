# Course 1 Notes

Specialization Order

* Neural Netowrks and Deeplearning 
* Improving DL 
* Structuring ML project
* Convolutional Neural Networks [often for images]
* NLP and Sequence Models [RNN and LSTM]

## What is a Neural Network? 

* ReLU: Recitfied Linear Unit has a function> 0 
* Standard NN: Use case for x -> y 
* CNN: Image data 
* RNN: Sequence data 

For Relu function, the gradient is equal to one. Makes gradient decsent faster and speeds up training on large data sets 

## Week 2

Image data is stored in your computer in 3 matricies corresponding to Red, Green and Blue. If your image is 64x64 pixels, then you will have 3x 64x64 matricies 

To unrow image data, create a feature vector of the image matricies 

### Logistic Regression 

sigmoid = 1/(1+e^-z)

z = regression function 

**Cost Function**

Need to find a global optimum for gradient loss - a convex shape is best 

Loss Function = L(pred_y, y) = (ylog(prep_y) + (1-y)log(1-pred_y))

Cost Function = 1/m * Sum(L(pred_y, y))

Loss function computes the error for a single observation, whereas the cost function computes the average of the loss across the observations 

**Gradient Descent** 

learning rate: how big of a step to find the minimum optimum 

Cost function should be convexed in order to have one minimal optima 

**Derivatives**

Derivative == slope 

Doesnt change on a straight line, but the slop varies for most cost functions 

change in y / change in x

derivative of a function finds the rate of change when you change the function input 

### Computation Graphs of Neural Networks 

An order of steps of computations for a given function. 

One step of backward propagation on a computation graph yeilds the derivaitive of the final output 

### Derivatives and computation graphs 

Finding the derivatives of each step in the computational graph reveals what change in step 1 affects other steps and the final output. Uses chain rule of calculus = the product of changes. 

When computing derivates, the easiest way is through backpropagation 

### Logistic Regression Gradient Descent 

This changes the weights and the intercept vectors using backpropagation 

w<sub>1</sub> = w<sub>1 </sub> - a*d(w<sub>1</sub>)

The weight decreases by the derivative times the learning rate 

### Gradient Descent Across All Observations 

The average of derivatives across the loss function 

### Vectorization & Broadcasting

Vecotirzation through numpy is faster

Braodcast = apply a mathical operation over columns 

### Cost function of logistic regression 


# quiz
What does a neuron compute? A neuron computes a linear function z = Wx +b followed by an activation function 

Logistic Loss = -(y(i)log(pred_y(i))) + (1-y(i))log(1-pred_y(i))

Numpy shape of a n<sub>x</sub> inputer features for observations X = [X<sub>1</sub> ... , X<sub>m</sub>] yeild sn array of (n<sub>x</sub>, m)







