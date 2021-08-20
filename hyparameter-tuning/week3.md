# Hyperparameter Tuning

## Tuning Process

Alpha (learning_rate) is the most important

Choose random set of points over grid search. Allows you to search over more values and find the important value sooner adn can refine the space of the grid to focus on 


1. use random search 

2. Implement a refine searching process to narrow down grid

## Using an Appropriate Scale to pick Hyperparameters


Random search does not mean a uniform search 

Sampling at random on a log scale of the learning rate 

```python

 r= np.random.rand() * -4 
 learning_rate = 10**r 
 # sample between a and b which is 10**a to 10**b [.00001, 1]
```
This spends the same level of resources uniformly for every value of r [-4, 1]

## pandas vs. cavier 

Babysitting one model: Forced to train one model and tune it (pandas model)

Training many models in parallel: Training many different types of models with different hyperparamters (cavier strategy)

Depends on computational resources 

## batch normalization 

Normalizing the inputs (features) X = (X-u)/sigma speeds up training 

Batch normalization normalizes Z so transformation of w and b are faster 

Normalize Z to have a mean of 0 and variance of 1 esp. if you are using the sigmoid function

Batch norm applied often with mini batch. Each Z is normalized only to that mini-batch 

If using batch norm, the parameter b (a constant in the Z function) is zeroed out 

## why does batch norm work

Makes weights in later layers more robust to changes to earlier layers

Covariate shift: The distribution of your inputs change 

So what batch norm does, is it reduces the amount that the distribution of these hidden unit values shifts around. The mean and variance stays at 0 and 1, rather than huge changes. 

Each mini batch is scaled by teh mean and variance scaled on just that mini batch. This adds noise to the Z within that mini batch which acts like a regularization effect. This is similiar to dropout. 

## batch norm at testing (predicting)

create an exponentially weighted average of mu and sigma to normalize Z at testing using a tuned beta and gamma 

## multiclass classfication with softmax

softmax regression: allows for learning of multiple classes 

The output layers equals the number of possible classes. Outputs the probability of each class. 

Using a softmax activation layer in the output layer, we can predict probability for each class given the inputs X 

Softmax: 

t = e**z(l)
a(i)(l) = t(i)/sum(t(i))

z = [5,2,-1,3]
t = [e**5, e**2, e**-1, e**3]
a = [e**5/sum(t), e**2/sum(t), ...]

Basically the normalization is the amount of each in the total, so they sum to 1. (probability)

hardmax: highest Z gets 1, others = 0 

softmax regression generalizes to logistic regression 

Loss function for softmax: 

L(y, pred_y) = -sum(Yj*log(pred_Yj) for all classes (j). This is negative 

This makes the class most likely to be the biggest prob 

Average all obervations across the dataset  is the loss for the entire model 

(1/m) * sum(L) of 1..m 

Gradient descent: dz = pred_y - y 

## deep learning frameworks

## tensorflow 

```python
import numpy as np
import tensorflow as tf

w = tf.Variable(0,dtype=tf.float32)
optimizer = tf.kera.optimizers.Adam(0.1) # learning rate

def train_step():
    with tf.GradientTape() as tape:
        cost = w**2 - 10*w + 25 # example cost function
    trainable_vars = [w]
    grads = tape.gradient(cost, traianble_vars)
    optimizer.apply_gradients(zip(grads, trainable_vars))

for i in range(10000):
    train_step()
print(w)

# w is what we want to optimize 

# new problem with X and Y data 

w = tf.Variable(0,dtype=tf.float32)
x = np.array([1.0, -10.0, 25.0])
optimizer = tf.kera.optimizers.Adam(0.1) # learning rate
def train(x,w, optimizer):
    def cost_func():
        return x[0] * w**2 + x[1] * w + x[2]
    for i in range(1000):
        optimizer.minimize(cost_func, [w])
    return w 

w = train(x, w, optimizer)

```

To find Beta for momementum between .9 and .99:
 r = np.random.rand() 
 beta = 1-10**(- r - 1)  











