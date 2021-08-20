# train, dev, test sets

training set: Used for your model to learn 

validation/dev set: Use to tune your ML

test/holdout set: offers unbiased estimate on how your ML is learning

# mismatched train and test datasets

Train and test sets come from the same source, and contain the same distribution

# bias vs variance

high bias: Underfitting and has too much noise in the learning to be something that solves the problem 

high variance: overfitting the training data. Lots of variance between error of train/test set 

## metrics to use 

train set error vs dev set error: 
    If train set error < dev set error = overfitting and has high variance 
    If train set error and dev set error are both high: high bias 
    If train set error is high and dev set error is higher: both high bias and high variance 
    If train set error = dev set error and both are low: low bias and low variance 

    Need to have a baseline of what is "good." What would a human error be? 

# basic recipe for machine learning 

If you have high bias -> bigger/better model (increasing more units), train longer, make netowrk deeper 

If you have high variance -> if you cannot generalize to the validation set (high validation error), get more data, regularize, simpler models, increase lamba 

# regularization

Prevents overfitting 

Regularize the W vector of weights. L2 regualrization = ||w**2|| = W.T*W 

L1 regularization: Compresses some Weights to 0. L2 is used much more often in practice 

Lambda: regularization parameter 

In aa Neural Network: 
    Frobenius Norm of the matrix (similar to L2)
    Weight Decay

# why does it prevent overfitting 

Weight decay will remove some nodes to simplify the model. A larger lambda sets weights to 0. 

It will turn the function closer to a logistic regression (for sigmoid activation and tanh because Z becomes small 

L2 is used the most 

Weight decay: a regualrization technique that results in gradient descent shrinking the weights on every iteration

Increasing lambda pushes weights closer to 0 

L2-regularization relies on the assumption that a model with small weights is simpler than a model with large weights. Thus, by penalizing the square values of the weights in the cost function you drive all the weights to smaller values. It becomes too costly for the cost to have large weights! This leads to a smoother model in which the output changes more slowly as the input changes.

What you should remember: the implications of L2-regularization on:

The cost computation:
A regularization term is added to the cost.
The backpropagation function:
There are extra terms in the gradients with respect to weight matrices.
Weights end up smaller ("weight decay"):
Weights are pushed to smaller values.

# Dropout regularization 

Randomly drops nodes to prevent overfitting and because it cant rely on any one feature. It will spread out weights. 

It will shrink the weights similar to L2 regularization

Implementing dropout with inverted dropout:

Do not do at testing (do not apply dropout and do not keep the 1/keep_prob factor)

You can adjust keep_prob to each layer so a lower keep_prob for high node layers you are worried about overfitting 

Increasing keep_prob reduces the regularization effect and causes the network to end with a lower training set error 

Note:

A common mistake when using dropout is to use it both in training and testing. You should use dropout (randomly eliminate nodes) only in training.
Deep learning frameworks like tensorflow, PaddlePaddle, keras or caffe come with a dropout layer implementation. Don't stress - you will soon learn some of these frameworks.
What you should remember about dropout:

Dropout is a regularization technique.
You only use dropout during training. Don't use dropout (randomly eliminate nodes) during test time.
Apply dropout both during forward and backward propagation.
During training time, divide each dropout layer by keep_prob to keep the same expected value for the activations. For example, if keep_prob is 0.5, then we will on average shut down half the nodes, so the output will be scaled by 0.5 since only the remaining half are contributing to the solution. Dividing by 0.5 is equivalent to multiplying by 2. Hence, the output now has the same expected value. You can check that this works even when keep_prob is other values than 0.5.

# other regularization methods

Data Augmentation: Changing an image slightly (flipping an image) or zooming into an image. This helps when getting more data is unfeasible 


Early Stopping: Stop training neural network once dev set error starts increasing 

# normalizing inputs 

normalization: (x-u)/variance 

Use the same mean and variance to scale training and test 

## why do we normalize 

If features are on very different scales, then weights vector will be on very different values. Noramlizing allows your cost function to be easier to optimize the minimum 

Necessary when features are on a different ranges 

# vanishing and exploding gradients 

pred_y = W^lW^l-1....W1*X 

A very deep neural network with lots of W, will explode pred_y 

If gradient descents are large, pred_y will be large 

If gradient descents are small, pred_y will be small and not able to learn anything 

TO cure, you need to initialize weights appropriatley. 


In a network of n hidden layers, n derivatives will be multiplied together. If the derivatives are large then the gradient will increase exponentially as we propagate down the model until they eventually explode, and this is what we call the problem of exploding gradient. Alternatively, if the derivatives are small then the gradient will decrease exponentially as we propagate through the model until it eventually vanishes, and this is the vanishing gradient problem.

In the case of exploding gradients, the accumulation of large derivatives results in the model being very unstable and incapable of effective learning, The large changes in the models weights creates a very unstable network, which at extreme values the weights become so large that is causes overflow resulting in NaN weight values of which can no longer be updated. On the other hand, the accumulation of small gradients results in a model that is incapable of learning meaningful insights since the weights and biases of the initial layers, which tends to learn the core features from the input data (X), will not be updated effectively. In the worst case scenario the gradient will be 0 which in turn will stop the network will stop further training

# weight initialization 

The larger n, the smaller w we want. Set W = np.random.randn(shape)* np.sqrt(1/n^l-1) (Xavier initialization)

If using relu, mutliply by np.sqrt(2/n^l-1)

# numerical approx. of gradients & gradient checking 

1. concatenate all W and B vectors into the shape called theta
2. Reshape all dw and db vectors into the shape called d_theta
3. Compute a approx of d_theta and make sure its close to d_theta

Dont use in training 

if alogorithm fails grad check, look at components to try to idenitfy bug 

Remember regularization 

Doesnt work with drop out

Run at random initialization 

What you should remember from this notebook:

Gradient checking verifies closeness between the gradients from backpropagation and the numerical approximation of the gradient (computed using forward propagation).
Gradient checking is slow, so you don't want to run it in every iteration of training. You would usually run it only to make sure your code is correct, then turn it off and use backprop for the actual learning process.