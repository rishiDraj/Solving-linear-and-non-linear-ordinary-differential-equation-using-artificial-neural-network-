import jax.numpy as np # importing the jax library
def sigmoid(x):
    return 1./(1. + np.exp(-x))

# Define a function for parameter initialization
def f(params, x):
    w0 = params[:8]
    b0 = params[8:16]
    w1 = params[16:24]
    b1 = params[25]
    x = sigmoid(x*w0 + b0)
    x = sigmoid(np.sum(x*w1) + b1)
    return x

# Initialize the parameters randomly
from jax import random
#pseudo-random number generator (PRNG) provided by JAX
key = random.PRNGKey(0)
params = random.normal(key, shape=(25,))


# Determine derivative of f with respect to x
from jax import grad

dfdx = grad(f,1)

grids = [11,101] # Defining the number of grid points to be considered

for i in grids:
    inputs = np.linspace(-2., 2., num=i)
    from jax import vmap
    f_vect = vmap(f, (None, 0)) # 0 indicates the mapped axis. It is column axis currently.
    dfdx_vect = vmap(dfdx, (None, 0))

    from jax import jit
    @jit
    def loss(params, inputs):
        eq = dfdx_vect(params, inputs) + 2.*inputs*f_vect(params, inputs)
        ic = f(params, 0.) - 1.
        # make it coveex so that the gradient can be determined efficiently.
        return np.mean(eq**2) + np.mean(ic**2)


    grad_loss = jit(grad(loss, 0))

    # Run the model
    epochs = 1000
    learning_rate = 0.1
    momentum = 0.99
    velocity = 0.
    for epoch in range(epochs):
        if epoch % 100 == 0:
            print('epoch: %3d loss: %.6f' % (epoch, loss(params, inputs)))
        gradient = grad_loss(params + momentum*velocity, inputs)
        velocity = momentum*velocity - learning_rate*gradient
        params += velocity


    # Plot
import matplotlib.pyplot as plt
plt.plot(inputs, np.exp(-inputs**2), label='exact')
plt.plot(inputs, f_vect(params, inputs), label='approx')

plt.legend()
plt.show()

# Define sigmoid function. It will be used in the neural network.

def sigmoid(x):
    return 1./(1. + np.exp(-x))

# Define a function for parameter initialization
def f1(params1, x):
    w0 = params1[:5]
    b0 = params1[5:10]
    w1 = params1[10:15]
    b1 = params1[16]
    x = sigmoid(x*w0 + b0)
    x = sigmoid(np.sum(x*w1) + b1)
    return x

key = random.PRNGKey(0)
params1 = random.normal(key, shape=(16,))

dfdx1 = grad(f1,1)

inputs1 = np.linspace(-2., 2., num=101)
from jax import vmap
f_vect1 = vmap(f1, (None, 0)) # 0 indicates the mapped axis. It is column axis currently.
dfdx_vect1 = vmap(dfdx1, (None, 0))

from jax import jit
@jit
def loss1(params1, inputs1):
    eq1 = dfdx_vect1(params1, inputs1) + 2.*inputs1*f_vect1(params1, inputs1)
    ic1 = f1(params1, 0.) - 1.
    # make it coveex so that the gradient can be determined efficiently.
    return np.mean(eq1**2) + np.mean(ic1**2)
  
grad_loss1 = jit(grad(loss1, 0))

  # Run the model
epochs = 1000
learning_rate = 0.1
momentum = 0.99
velocity = 0.
for epoch in range(epochs):
    if epoch % 100 == 0:
        print('epoch: %3d loss: %.6f' % (epoch, loss1(params1, inputs1)))
        gradient = grad_loss1(params1 + momentum*velocity, inputs1)
        velocity = momentum*velocity - learning_rate*gradient
        params1 += velocity


  # Plot
import matplotlib.pyplot as plt
plt.plot(inputs1, np.exp(-inputs1**2), label='exact')
plt.plot(inputs1, f_vect1(params1, inputs1), label='approx')
plt.legend()
plt.show()

# Define sigmoid function. It will be used in the neural network.

def sigmoid(x):
    return 1./(1. + np.exp(-x))

# Define a function for parameter initialization
def f1(params1, x):
    w0 = params1[:20]
    b0 = params1[20:40]
    w1 = params1[40:60]
    b1 = params1[61]
    x = sigmoid(x*w0 + b0)
    x = sigmoid(np.sum(x*w1) + b1)
    return x

key = random.PRNGKey(0)
params1 = random.normal(key, shape=(61,))

dfdx1 = grad(f1,1)

inputs1 = np.linspace(-2., 2., num=101)
from jax import vmap
f_vect1 = vmap(f1, (None, 0)) # 0 indicates the mapped axis. It is column axis currently.
dfdx_vect1 = vmap(dfdx1, (None, 0))

from jax import jit
@jit
def loss1(params1, inputs1):
    eq1 = dfdx_vect1(params1, inputs1) + 2.*inputs1*f_vect1(params1, inputs1)
    ic1 = f1(params1, 0.) - 1.
    # make it coveex so that the gradient can be determined efficiently.
    return np.mean(eq1**2) + np.mean(ic1**2)

grad_loss1 = jit(grad(loss1, 0))

  # Run the model
epochs = 1000
learning_rate = 0.1
momentum = 0.99
velocity = 0.
for epoch in range(epochs):
    if epoch % 100 == 0:
        print('epoch: %3d loss: %.6f' % (epoch, loss1(params1, inputs1)))
        gradient = grad_loss1(params1 + momentum*velocity, inputs1)
        velocity = momentum*velocity - learning_rate*gradient
        params1 += velocity

  
  # Plot
import matplotlib.pyplot as plt
plt.plot(inputs1, np.exp(-inputs1**2), label='exact')
plt.plot(inputs1, f_vect1(params1, inputs1), label='approx')
plt.legend()
plt.show()



# Define sigmoid function. It will be used in the neural network.

def sigmoid(x):
    return 1./(1. + np.exp(-x))

# Define a function for parameter initialization
def f1(params1, x):
    w0 = params1[:8]
    b0 = params1[8:16]
    w1 = params1[16:24]
    b1 = params1[25]
    x = sigmoid(x*w0 + b0)
    x = sigmoid(np.sum(x*w1) + b1)
    return x

key = random.PRNGKey(0)
params1 = random.normal(key, shape=(25,))

dfdx1 = grad(f1,1)

inputs1 = np.linspace(-2., 2., num=101)
from jax import vmap
f_vect1 = vmap(f1, (None, 0)) # 0 indicates the mapped axis. It is column axis currently.
dfdx_vect1 = vmap(dfdx1, (None, 0))

from jax import jit
@jit
def loss1(params1, inputs1):
    eq1 = dfdx_vect1(params1, inputs1) + 2.*inputs1*f_vect1(params1, inputs1)
    ic1 = f1(params1, 0.) - 1.
    # make it coveex so that the gradient can be determined efficiently.
    return np.mean(eq1**2) + np.mean(ic1**2)

grad_loss1 = jit(grad(loss1, 0))

  # Run the model
epochs = 1000
learning_rate = 0.1
momentum = 0.99
velocity = 0.
for epoch in range(epochs):
    if epoch % 100 == 0:
        print('epoch: %3d loss: %.6f' % (epoch, loss1(params1, inputs1)))
        gradient = grad_loss1(params1 + momentum*velocity, inputs1)
        velocity = momentum*velocity - learning_rate*gradient
        params1 += velocity

  
  # Plot
import matplotlib.pyplot as plt
plt.plot(inputs1, np.exp(-inputs1**2), label='exact')
plt.plot(inputs1, f_vect1(params1, inputs1), label='approx')
plt.legend()
plt.show()

# Define sigmoid function. It will be used in the neural network.

def sigmoid(x):
    return 1./(1. + np.exp(-x))

# Define a function for parameter initialization
def f1(params1, x):
    w0 = params1[:8]
    b0 = params1[8:16]
    w1 = params1[16:24]
    b1 = params1[25]
    x = sigmoid(x*w0 + b0)
    x = sigmoid(np.sum(x*w1) + b1)
    return x

key = random.PRNGKey(0)
params1 = random.normal(key, shape=(25,))

dfdx1 = grad(f1,1)

inputs1 = np.linspace(-2., 2., num=101)
from jax import vmap
f_vect1 = vmap(f1, (None, 0)) # 0 indicates the mapped axis. It is column axis currently.
dfdx_vect1 = vmap(dfdx1, (None, 0))

from jax import jit
@jit
def loss1(params1, inputs1):
    eq1 = dfdx_vect1(params1, inputs1) + 2.*inputs1*f_vect1(params1, inputs1)
    ic1 = f1(params1, 0.) - 1.
    # make it coveex so that the gradient can be determined efficiently.
    return np.mean(eq1**2) + np.mean(ic1**2)

grad_loss1 = jit(grad(loss1, 0))

  # Run the model
epochs = 1000
learning_rate = 0.1
momentum = 0.99
velocity = 0.
for epoch in range(epochs):
    if epoch % 100 == 0:
        print('epoch: %3d loss: %.6f' % (epoch, loss1(params1, inputs1)))
        gradient = grad_loss1(params1 + momentum*velocity, inputs1)
        velocity = momentum*velocity - learning_rate*gradient
        params1 += velocity

  
  # Plot
import matplotlib.pyplot as plt
plt.plot(inputs1, np.exp(-inputs1**2), label='exact')
plt.plot(inputs1, f_vect1(params1, inputs1), label='approx')
plt.legend()
plt.show()



# Define sigmoid function. It will be used in the neural network.

def sigmoid(x):
    return 1./(1. + np.exp(-x))

# Define a function for parameter initialization
def f1(params1, x):
    w0 = params1[:8]
    b0 = params1[8:16]
    w1 = params1[16:24]
    b1 = params1[25]
    x = sigmoid(x*w0 + b0)
    x = sigmoid(np.sum(x*w1) + b1)
    return x

key = random.PRNGKey(0)
params1 = random.normal(key, shape=(25,))

dfdx1 = grad(f1,1)

inputs1 = np.linspace(-20., 20., num=101)
from jax import vmap
f_vect1 = vmap(f1, (None, 0)) # 0 indicates the mapped axis. It is column axis currently.
dfdx_vect1 = vmap(dfdx1, (None, 0))

from jax import jit
@jit
def loss1(params1, inputs1):
    eq1 = dfdx_vect1(params1, inputs1) + 2.*inputs1*f_vect1(params1, inputs1)
    ic1 = f1(params1, 0.) - 1.
    # make it coveex so that the gradient can be determined efficiently.
    return np.mean(eq1**2) + np.mean(ic1**2)
  
grad_loss1 = jit(grad(loss1, 0))

  # Run the model
epochs = 1000
learning_rate = 0.1
momentum = 0.99
velocity = 0.
for epoch in range(epochs):
    if epoch % 100 == 0:
        print('epoch: %3d loss: %.6f' % (epoch, loss1(params1, inputs1)))
        gradient = grad_loss1(params1 + momentum*velocity, inputs1)
        velocity = momentum*velocity - learning_rate*gradient
        params1 += velocity

  
  # Plot
import matplotlib.pyplot as plt
plt.plot(inputs1, np.exp(-inputs1**2), label='exact')
plt.plot(inputs1, f_vect1(params1, inputs1), label='approx')
plt.legend()
plt.show()


#-----------------------------------------------------

import autograd.numpy as np
from autograd import grad, elementwise_grad
import autograd.numpy.random as npr
from autograd.misc.optimizers import adam
import time


# Define a sigmoid activation function. Can also be used from library. This is a simple definition.
def sigmoid_activation(x):
 
    return x / (1.0 + np.exp(-x))
    
    
    
def init_random_params(layer_sizes):
    rs=npr.RandomState(0)
#Define a list of (weights, biases tuples, one for each layer."
    return [(rs.randn(insize, outsize),   # weight matrix
             rs.randn(outsize))           # bias vector
            for insize, outsize in zip(layer_sizes[:-1], layer_sizes[1:])]
# The above line will run the for loop from insize to outsize, and will store the values 
# layer_sizes[:-1] fills the weight matrix
# layer_sizes[1:] fills the bias array 

# Define function y based on neural networks. Outputs are linearly related to biases and weights.
# Outputs of one layer are used as inputs to another layer via activation function.
def y(params, inputs):
    "Neural network functions"
    for W, b in params:
        outputs = np.dot(inputs, W) + b
        inputs = sigmoid_activation(outputs)    
    return outputs
    
# initial guess for params:
params = init_random_params(layer_sizes=[1, 10, 1])

layer_sizes=[1, 8, 1]
print(layer_sizes[1:])


dydx = elementwise_grad(y, 1) # partial derivative of y with respect to inputs i.e. x



y0 = 1.0
x = np.linspace(-2, 2).reshape((-1, 1))



# Define the objective function.
def lossfunction(params,step):
    # The objective is to minimize to zero.
    # dydx = -2xy
#    ycall = y(params,inputs)
    zeq = dydx(params, x) + (2*x**3)+np.exp(-x)
    y0 = 1.0
    ic = y(params, 0) - y0 # For my solution i.e. a set of paramaters 'params' this condition should be satisfied
    # since this is the intial condition.
    # If I minimize zeq and ic together or in some combined form, I will get a set of 'params' that give me
    # solution of dy/dx
    # Let us setup the loss function as zeq + ic
    return np.mean(zeq**2 + ic**2)

def callback(params,step, g):
    if step % 100 == 0:
        print("Iteration {0:3d} lossfunction {1}".format(step,lossfunction(params,step)))
        

#ODE solver for 8 nodes
# grad(losfunciton) = d J(theta) / d theta
params = adam(grad(lossfunction), params, callback=callback, step_size=0.1, num_iters=1000)

#Plot for 8 nodes
tfit = np.linspace(-2, 2).reshape(-1, 1)
import matplotlib.pyplot as plt
plt.plot(tfit, y(params, tfit), label='soln') 
plt.plot(tfit,(((-tfit**4)/2)+np.exp(-tfit)), 'r--', label='analytical soln')
plt.legend()
plt.xlabel('x')
plt.ylabel('$y$')
plt.xlim([-2, 2])
plt.savefig('odenn.png')