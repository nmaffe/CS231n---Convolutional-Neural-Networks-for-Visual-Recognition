from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights. # sembra che f(X,W) = X*W e non W*X
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    dW = np.zeros(W.shape) # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in range(num_train): # loop over the training images to sum L_i
        
        scores = X[i].dot(W) # scores (1, C)
        
        correct_class_score = scores[y[i]] # NB the class corresponds with the index of vector y
        
        for j in range(num_classes): # loop over the classes to compute L_i
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1 # note delta = 1
            if margin > 0:
                loss += margin
                dW[:, j] += X[i]#maffe
                dW[:, y[i]] -= X[i]#maffe

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train
    dW /= num_train

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)
    dW += reg * 2* W

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #pass
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    return loss, dW



def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape) # initialize the gradient as zero
    num_train = X.shape[0] #maffe
    num_classes = W.shape[1] #maffe

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    S = np.dot(X,W)
    
    correct_class_scores = S[np.arange(num_train), y] #(N,)
    correct_class_scores = correct_class_scores.reshape(-1, 1) # column vector
    
    margins = S - correct_class_scores + 1 #maffe
    #mask_margins = np.ma.MaskedArray(margins, mask=(margins<1)) #delta=1
    #mask_margins = np.where(margins<0, 0, margins)#delta=1 #maffe
    
    margins = np.maximum(margins, 0.0)
    
    # Set to zero the values where j=yi
    margins[np.arange(num_train), y] = 0.0
    
    loss = np.sum(margins)
    
    loss /= float(num_train)
    
    loss += reg * np.sum(W * W)
    
    
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # mask: there is 1 associated to wrong class j if margin>0 and 0 associated to true class y_1 (since there margin has been set to zero) 
    grad_mask = (margins > 0).astype(int)
    
    # in association to the elements where there is the true class i plug the sum of the wrong classes j where margin>0
    # the sum is along the columns, hence over the classes j
    grad_mask[np.arange(y.shape[0]), y] = - np.sum(grad_mask, axis=1)
    #print(grad_mask[5])
    
    # At this stage I have a mask with 1 at the elements corresponding to the wrong class if margin>0, while at the element corresponding to the true class (y) there is a value with is the opposite of the sum of the number of classes j where margin>0
    
    dW = np.dot(X.T, grad_mask) 

    dW /= float(num_train)
    dW += reg * 2 * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
