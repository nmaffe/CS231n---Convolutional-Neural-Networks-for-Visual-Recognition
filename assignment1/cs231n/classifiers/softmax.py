from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_classes = W.shape[1]
    num_train = X.shape[0]
    scores = np.dot(X, W)
    
    # Softmax Loss
    for i in range(num_train):
        
        # i-th image
        f_i = scores[i]
        softmax_i = np.exp(f_i)/np.sum(np.exp(f_i))
        loss += -np.log(softmax_i[y[i]])
        
        # Gradient of softmax loss
        for j in range(num_classes):
            dW[:,j] += X[i] * softmax_i[j] 
        dW[:,y[i]] -= X[i,:]
        

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    loss = loss/num_train + reg * np.sum(W * W)
    
    dW = dW/num_train + 2 * reg * W
    
    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    num_classes = W.shape[1]
    num_train = X.shape[0]
    scores = np.dot(X,W) #scores is matrix (N,C)
    
    # Softmax Loss
    norm = np.sum(np.exp(scores), axis=1) # vector of normalizations
    softmax = np.exp(scores)/norm.reshape(num_train, -1)  #softmax it's a matrix, same shape as scores matrix
    
    loss = np.sum(-np.log(softmax[np.arange(num_train), y])) # softmax loss is a scalar
    
    # Gradient of softmax loss
    softmax[np.arange(num_train),y] -= 1
    dW = np.dot(X.T, softmax)
       
    
    # Average and regularization
    loss = loss/num_train + reg * np.sum(W * W)
    dW = dW/num_train + 2 * reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
