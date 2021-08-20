# mini batch gradient descent 

Mini batch: break up training dataset of M examples into m mini batches (5M records to 5K records of 1000 mini batches)

X {t}, Y {t}: t denotes a mini batch 

Implemenation: Loop through each minibatch and inside the loop implement one step of gradient descent 

    - forward prop of X{t} from l to L
    - activate(z) from l to L
    - compute cost function with relative to number of observations in batch t 
    - compute backprop
    - update weights and b parameters 

This is also known as an epoch. Runs much faster than a batch version

The cost curve will be noiser with mini batch than batch, but trend should still be zero

Choosing the size of the minibatch, 

If minibatch # of observations = 1, then this is stochastic gradient descent. This will cause the cost function to be incredibly noisy. 

In practice, choose a reasonable size for each minibatch 

A huge con to stochastic gradient descent is the loss of time speed up because you are no longer using vectorization

Batch takes too long

Guidelines: If your training set is small, use batch gradient descent. Typical mini batch sizes that are a power of 2: 64, 128, 256, 512, 1024 etc. 

This can be researched and implemented in your hyperparameter grid search 

# exponentially weighted averages 

V(t) = .9*V(t-1) + .1*Y(t)
V(t) = B*V(t-1) * (1-B)*Y(t)

.9 = B, the average of the most recent (1/(1-B)) for each grain level ( in his example, days)

This smoothes a noisy data value, for example, temperatures over time.

V(t) is the average over (1/1-B)

B ^ (1/(1-B)) ~ 1/e

```python 
v = 0 
v = Bv + (1-B)y
```

# bias correction of the weighted average 

Divide V(t) by (1-B^t)

This will help when t is small and not too biased too recent. For example, only on the first two or three days. 

# gradient descent with momentum 

Faster than gradient descent 

Smooothes out gradient steps with weighted average. Deriviative terms are like accelreation, while weighted averge is velocity 

Implementation requires an alpha (learning rate) and beta term (relating to the past (1/1-B) gradients)

on iteration t:
1. compute dW, dB
2. Vdw = B*Vdw + (1-B)*dW; Vdb - B*Vdb + (1-B)*db 
3. W = W - a*Vdw; b = b - a*Vdb

# RMS Prop

Root Mean Squared Prop 

V also known as S here:

on iteration t:
1. compute dW, dB
2. Vdw = B*Vdw + (1-B)*dW^2; Vdb - B*Vdb + (1-B)*db^2 
3. W = W - a*(dw/ sqrt(Vdw)); b = b - a*(db/sqrt(Vdb))

Allows you to learn a larger learning rate alpha 

# Adam Optimization 

Optimziation methods dont generalize well. 

Implementation: 

1. set parameters to 0
2. on iteration t:
    - compute dw, db using current batch or mini batch 
    - Vdw and Vdb from momentum 
    - Sdw and Sdb from RMS prop
    - Correct Vdw, Vdb, Sdw, Ddb by dividing by (1-B^t)
    - W = W - a*(Vdw/sqrt(Sdw); b = b - a*(Vdb/sqrt(Sdb)))
    - add small e to sqrt so no sqrt of 0 errors

Hyperparamters: 
a: tuned
B1 = .9 
B2 = .999
e = 10^-8



# Learning Rate Decay

Slowly decreasing learning rate overt ime to speed up learning 

A fixed rate of alpha might hurt learning and finding optimal cost

Start large and then minimize over epochs will take smaller steps around the optima 

a_t = (1/(1 + decay_rate*epoch_num))*alpha_zero

exponential decay: 

alpha_t = .95^epoch_num * alpha_zero

# The problem of local optima 

Most points of zero gradient are saddle points

Plateaus in the "saddle" can slow down learning 

Unlikely to get stuck in a bad local optima 




More Notes:

Note that:

The velocity is initialized with zeros. So the algorithm will take a few iterations to "build up" velocity and start to take bigger steps.
If  𝛽=0 , then this just becomes standard gradient descent without momentum.
How do you choose  𝛽 ?

The larger the momentum  𝛽  is, the smoother the update, because it takes the past gradients into account more. But if  𝛽  is too big, it could also smooth out the updates too much.
Common values for  𝛽  range from 0.8 to 0.999. If you don't feel inclined to tune this,  𝛽=0.9  is often a reasonable default.
Tuning the optimal  𝛽  for your model might require trying several values to see what works best in terms of reducing the value of the cost function  𝐽 .

What you should remember:

Momentum takes past gradients into account to smooth out the steps of gradient descent. It can be applied with batch gradient descent, mini-batch gradient descent or stochastic gradient descent.
You have to tune a momentum hyperparameter  𝛽  and a learning rate  𝛼 .

