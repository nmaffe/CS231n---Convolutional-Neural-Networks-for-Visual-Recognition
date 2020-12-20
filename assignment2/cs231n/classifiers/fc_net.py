from builtins import range
from builtins import object
import numpy as np

from cs231n.layers import *
from cs231n.layer_utils import *


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg

        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian centered at 0.0 with               #
        # standard deviation equal to weight_scale, and biases should be           #
        # initialized to zero. All weights and biases should be stored in the      #
        # dictionary self.params, with first layer weights                         #
        # and biases using the keys 'W1' and 'b1' and second layer                 #
        # weights and biases using the keys 'W2' and 'b2'.                         #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        self.params['W1'] = np.random.randn(input_dim, hidden_dim) * weight_scale
        self.params['b1'] = np.zeros(hidden_dim)
        self.params['W2'] = np.random.randn(hidden_dim, num_classes) * weight_scale
        self.params['b2'] = np.zeros(num_classes)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################


    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the two-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        
        # Unpack variables from the params dictionary
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        #reg = self.reg
        
        #print('reg here: ', reg)
            
        #N = X.shape[0]
        #X = X.reshape(N, np.prod(X.shape[1:]))
        
        # First affine layer
        #z = np.dot(X, W1) + b1
        
        # Non linearity (relu)
        #h1 = np.maximum(0, z)
        
        # Second affine layer
        #h2 = np.dot(h1, W2)
                
        #scores = h2 + b2
        
        """ Instead let's use the convenient modules"""
        # Affine + RELU
        h1, h1_cache = affine_relu_forward(X, W1, b1)
        # Affine forward
        scores, scores_cache = affine_forward(h1, W2, b2)
        
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the two-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization!                   #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #pass
        """
        Manually by computing backprop
        # Softmax loss
        N = X.shape[0]
        exp_scores = np.exp(scores) #(N,C)
        num = exp_scores[np.arange(N), y] #(N,)
        den = np.sum(exp_scores, axis=1) #(N,)
                
        L = num / den #(N,)
        L = -np.log(L)
        L = np.sum(L, axis=0)/N + 0.5*reg*np.sum(W1*W1) + 0.5*reg*np.sum(W2*W2)
        
        loss = L
        
        # Gradients dW1, dW2, db1, db2   
        
        # dscores resta un po un mistero
        dscores = np.exp(scores) / np.sum(exp_scores, axis=1).reshape(N, -1) #(N,C)
        dscores[np.arange(N), y] -= 1.0
        dscores /= float(N) # (N,C)
                          
        
        # dh2
        dh2 = dscores
        
        # db2
        db2 = np.sum(dscores, axis=0)
        
        # dW2
        dW2 = np.dot(h1.T, dh2) + reg * W2
        
        # dh1
        dh1 = np.dot(dh2, W2.T)
        
        # dz (relu backward)
        dh1_dz = 1*(z>0)
        dz = dh1*dh1_dz
        
        # db1
        db1 = np.sum(dz, axis=0)
        
        # dW1
        dW1 = np.dot(X.T, dz) + reg * W1
        """
        
        
        """ Instead let's use the convenient modules"""
        # softmax_loss(x, y)
        loss, dscores = softmax_loss(scores, y)
        loss += 0.5 * self.reg * (np.sum(W1*W1) + np.sum(W2*W2))
        
        # affine_backward(dout, cache)
        dh1, dW2, db2 = affine_backward(dscores, scores_cache)
        dW2 += self.reg * W2
        
        # affine_relu_backward(dout, cache)
        dx, dW1, db1 = affine_relu_backward(dh1, h1_cache)
        dW1 += self.reg * W1
        
        # Store
        grads['W1'], grads['W2'] = dW1, dW2
        grads['b1'], grads['b2'] = db1, db2

        
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads

# Additions to original code    
def W_key(layer_idx):
    return "W" + str(layer_idx)


def b_key(layer_idx):
    return "b" + str(layer_idx)


def gamma_key(layer_idx):
    return "gamma" + str(layer_idx)


def beta_key(layer_idx):
    return "beta" + str(layer_idx)
    

# Additions to original code    
def affine_batchnorm_relu_forward(x, w, b, gamma, beta, bn_param):
    
    out_affine, cache_affine = affine_forward(x, w, b)
        
    out_bn, cache_bn = batchnorm_forward(out_affine, gamma, beta, bn_param)
    
    out_relu, cache_relu = relu_forward(out_bn)
    
    out_cache = cache_affine, cache_bn, cache_relu
    
    return out_relu, out_cache
    
# Additions to original code        
def affine_batchnorm_relu_backward(dout, cache):
    
    # unpack the cache
    cache_affine, cache_bn, cache_relu = cache
    
    # relu backward
    dout = relu_backward(dout, cache_relu)
    
    # batchnorm backward
    dout, dgamma, dbeta = batchnorm_backward_alt(dout, cache_bn)
    
    # affine backward
    dx, dw, db = affine_backward(dout, cache_affine)
    
    # it has to return something like this
    return dx, dw, db, dgamma, dbeta
    

# Additions to original code    
def affine_layernorm_relu_forward(x, w, b, gamma, beta, bn_param):
    
    out_affine, cache_affine = affine_forward(x, w, b)
        
    out_bn, cache_bn = layernorm_forward(out_affine, gamma, beta, bn_param)
    
    out_relu, cache_relu = relu_forward(out_bn)
    
    out_cache = cache_affine, cache_bn, cache_relu
    
    return out_relu, out_cache
    
    
# Additions to original code        
def affine_layernorm_relu_backward(dout, cache):
    
    # unpack the cache
    cache_affine, cache_bn, cache_relu = cache
    
    # relu backward
    dout = relu_backward(dout, cache_relu)
    
    # batchnorm backward
    dout, dgamma, dbeta = layernorm_backward(dout, cache_bn)
    
    # affine backward
    dx, dw, db = affine_backward(dout, cache_affine)
    
    # it has to return something like this
    return dx, dw, db, dgamma, dbeta
    
    
class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    dropout and batch/layer normalization as options. For a network with L layers,
    the architecture will be

    {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch/layer normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
                 dropout=1, normalization=None, reg=0.0,
                 weight_scale=1e-2, dtype=np.float32, seed=None):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=1 then
          the network should not use dropout at all.
        - normalization: What type of normalization the network should use. Valid values
          are "batchnorm", "layernorm", or None for no normalization (the default).
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deteriminstic so we can gradient check the
          model.
        """
        self.normalization = normalization
        self.use_dropout = dropout != 1
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}
        
        # Addition wrt original code
        # Compose a list of lists. Every sublist is a certain layer type with or without dropout
        # The last layer is an affine layer
        self.layers = []
        
        for l in range(self.num_layers-1):
            
            if self.normalization=="batchnorm":
                layer = ["affine_batchnorm_relu"]
                
            elif self.normalization=="layernorm":
                layer = ["affine_layernorm_relu"]
                
            else:
                layer = ["affine_relu"]

            if dropout != 1:
                layer.append("dropout")
                
            self.layers.append(layer)
            
        # last layer is a affine
        self.layers.append(["affine"])
        
        print("Network structure", self.layers)
    
    

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution centered at 0 with standard       #
        # deviation equal to weight_scale. Biases should be initialized to zero.   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to ones and shift     #
        # parameters should be initialized to zeros.                               #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        
        D = input_dim
        C = num_classes
        dims = [D] + hidden_dims + [C] # [D, H1, H2, C]
        
        for l in range(1, len(dims)):
            # Dimensions of W matrix of current layer
            n1, n2 = dims[l-1:l+1]
            # Initialize Ws and bs
            self.params[W_key(l)] = np.random.randn(n1, n2) * weight_scale
            self.params[b_key(l)] = np.zeros(n2)
        
        # Initialize gammas and betas for bn
        if self.normalization in ("batchnorm", "layernorm"):
            for l in range(1, len(dims)-1):
                n = dims[l]
                self.params[gamma_key(l)] = np.ones(n)
                self.params[beta_key(l)] = np.zeros(n)
                
        
        
        

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.normalization=='batchnorm':
            self.bn_params = [{'mode': 'train'} for i in range(self.num_layers - 1)]
        if self.normalization=='layernorm':
            self.bn_params = [{} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)
    
    

    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.

        Input / output: Same as TwoLayerNet above.
        """
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param['mode'] = mode
        if self.normalization=='batchnorm':
            for bn_param in self.bn_params:
                bn_param['mode'] = mode
                
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully-connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #print('BEGIN FORWARD ...')

        
        cache = {}
        
        out = X
        layers = self.layers # Get list of layers
        #print('here forward to the NN ', layers)
        
        # All layers except the last one
        for l, layer in enumerate(layers[:-1], 1):
            
            
            if 'affine' in layer:              
                W, b = self.params[W_key(l)], self.params[b_key(l)]
                out, out_cache = affine_forward(out, W, b)
                          
            elif 'affine_relu' in layer:
                
                W, b = self.params[W_key(l)], self.params[b_key(l)]
                out, out_cache = affine_relu_forward(out, W, b)
                                
            elif 'affine_batchnorm_relu' in layer:
                
                W, b = self.params[W_key(l)], self.params[b_key(l)]
                gamma, beta = self.params[gamma_key(l)], self.params[beta_key(l)]
                bn_param = self.bn_params[l-1]                
                out, out_cache = affine_batchnorm_relu_forward(out, W, b, gamma, beta, bn_param)
                
            elif 'affine_layernorm_relu' in layer:
                
                W, b = self.params[W_key(l)], self.params[b_key(l)]
                gamma, beta = self.params[gamma_key(l)], self.params[beta_key(l)]
                bn_param = self.bn_params[l-1]                
                out, out_cache = affine_layernorm_relu_forward(out, W, b, gamma, beta, bn_param)
            
            else: 
                print("WARNING: No layer found during forward", layer)
                
            if 'dropout' in layer:
                    out, out_cache_dropout = dropout_forward(out, self.dropout_param)
                    out_cache = out_cache, out_cache_dropout
                  
            
            # Cache current layer
            # e.g. cache['1'] will be cache for 1st layer
            cache[str(l)] = out_cache
            
                    
        # Last layer which is affine
        l = self.num_layers
        W, b = self.params[W_key(l)], self.params[b_key(l)]
        scores, scores_cache = affine_forward(out, W, b) 
        
            
        #print('END FORWARD ...')    
                           

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early
        if mode == 'test':
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully-connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch/layer normalization, you don't need to regularize the scale   #
        # and shift parameters.                                                    #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****    
        
        # softmax_loss(x, y)
        loss, dscores = softmax_loss(scores, y)
        
        # Regul the loss 
        weight_sums = np.sum(
            np.sum(self.params[W_key(l)] ** 2) for l in range(1, self.num_layers+1)
        )
        loss += 0.5 * self.reg * (weight_sums)
        
        
        # First backward is affine_backward    
        l = self.num_layers
        dout, dW, db = affine_backward(dscores, scores_cache)
        dW += self.reg * self.params[W_key(l)]
        grads.update({ W_key(l): dW, b_key(l): db })

        # Loop through all previous layers
        for l, layer in reversed(list(enumerate(self.layers[:-1], 1))):
            
            # Get layer cache
            layer_cache = cache[str(l)]
            
            # In case of dropout, we will be in the case of a layer composed of
            # layer = ["some_layer_type", "dropout"]
            if "dropout" in layer:
                
                    # In such case the forward cache will be made by two caches
                    out_cache, out_cache_dropout = layer_cache
                    
                    # calculate backward for Dropout
                    dout = dropout_backward(dout, out_cache_dropout)
                    
                    # Set the layer cache to the cache of the some_layer_type
                    layer_cache = out_cache
            
            if "affine" in layer:
                dout, dw, db = affine_backward(dout, layer_cache)
            
            elif "affine_relu" in layer:
                dout, dW, db = affine_relu_backward(dout, layer_cache)
                dW += self.reg * self.params[W_key(l)]
                grads.update({ W_key(l): dW, b_key(l): db })

            elif "affine_batchnorm_relu" in layer:
                dout, dW, db, dgamma, dbeta = affine_batchnorm_relu_backward(dout, layer_cache)
                dW += self.reg * self.params[W_key(l)]
                grads.update({W_key(l): dW, b_key(l): db,gamma_key(l): dgamma, beta_key(l): dbeta})
                
            elif "affine_layernorm_relu" in layer:
                dout, dW, db, dgamma, dbeta = affine_layernorm_relu_backward(dout, layer_cache)
                dW += self.reg * self.params[W_key(l)]
                grads.update({W_key(l): dW, b_key(l): db,gamma_key(l): dgamma, beta_key(l): dbeta})

            
            else: print("WARNING: No layer found during backward", layer)
        
        
        
 
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        return loss, grads
