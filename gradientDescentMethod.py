
import numpy as np
#from numpy import *
#import numpy.linalg as linalg
#import scipy.optimize as optimize
def loss(x, w, t):
    N, D = np.shape(x)
    y = np.matmul(x,w)
    loss = (y - t)
    return loss

def cost(x,w, t):
    '''
    Evaluate the cost function in a vectorized manner for 
    inputs `x` and targets `t`, at weights `w1`, `w2` and `b`.
    '''
    N, D = np.shape(x)
    return (loss(x, w,t) **2).sum() / (2.0 * 5)
def getGradient(x, w, t):
    N, D = np.shape(x)
    gradient = (2) * np.matmul(x.T, loss(x,w,t))
    return gradient

def getGradient(x, w, t):
    N, D = np.shape(x)
    gradient = (1.0/ float(N)) * np.matmul(np.transpose(x), loss(x,w,t))
    return gradient


def gradientDescentMethod(x, t, alpha=0.1, tolerance=1e-2):
    N, D = np.shape(x)
    #w = np.random.randn(D)
    w = np.zeros([D])
    # Perform Gradient Descent
    iterations = 1
    w_cost = [(w, cost(x,w, t))]
    while True:
        dw = getGradient(x, w, t)
        w_k = w - alpha * dw
        w_cost.append((w, cost(x, w, t)))
        # Stopping Condition
        if np.sum(abs(w_k - w)) < tolerance:
            print ("Converged.")
            break
        if iterations % 100 == 0:
            print ("Iteration: %d - cost: %.4f" %(iterations, cost(x, w, t)))
        iterations += 1
        w = w_k
    return  w, w_cost
x = np.array([[1,1,1,1,1],[1,2,4,8,16],[1,3,9,27,81],[1,4,16,64,256],[1,5,25,125,625]])
t = np.array([[5],[31],[121],[341],[781]])
print(gradientDescentMethod(x,t))