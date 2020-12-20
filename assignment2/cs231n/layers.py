from builtins import range
import numpy as np


def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None
    ###########################################################################
    # TODO: Implement the affine forward pass. Store the result in out. You   #
    # will need to reshape the input into rows.                               #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    N = x.shape[0]
    other_dims = x.shape[1:]
    num_other_dims = np.prod(other_dims) 
    x2D = x.reshape(N, num_other_dims)
    
    out = np.dot(x2D, w) + b

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)
      - b: Biases, of shape (M,)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the affine backward pass.                               #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    # Flatten x into (N, D)
    N = x.shape[0]
    D = np.prod(x.shape[1:]) 
    x2D = x.reshape(N, D)

    # Gradient db
    db = np.sum(dout, axis=0)
    dz = dout
    
    # Gradient dx reshaped as the original x shape, e.g. (N, d1, ..., d_k)
    dx = np.dot(dout, w.T).reshape(x.shape)
       
    # Gradient dw    
    dw = np.dot(x2D.T, dout)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    ###########################################################################
    # TODO: Implement the ReLU forward pass.                                  #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    out = np.maximum(x, 0.0)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = x
    #print('inside relu_forward', cache.shape)
    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    ###########################################################################
    # TODO: Implement the ReLU backward pass.                                 #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    # dout/dx = 0 if x<0, = 1 if x>0. dx = dout * dout/dx
    
    #dout[x<0] = 0
    #dx = dout
    #print('siamo in relu_backward ', dout[0].shape, dout[1].shape, dout[2].shape, x.shape)
    # NB dout deve essere un singolo ndarray chiaramente
    dx = dout *  (x > 0).astype(int)
    
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def batchnorm_forward(x, gamma, beta, bn_param):
    """
    Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7
    implementation of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)

    N, D = x.shape
    running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

    out, cache = None, None 
    if mode == 'train':
        #######################################################################
        # TODO: Implement the training-time forward pass for batch norm.      #
        # Use minibatch statistics to compute the mean and variance, use      #
        # these statistics to normalize the incoming data, and scale and      #
        # shift the normalized data using gamma and beta.                     #
        #                                                                     #
        # You should store the output in the variable out. Any intermediates  #
        # that you need for the backward pass should be stored in the cache   #
        # variable.                                                           #
        #                                                                     #
        # You should also use your computed sample mean and variance together #
        # with the momentum variable to update the running mean and running   #
        # variance, storing your result in the running_mean and running_var   #
        # variables.                                                          #
        #                                                                     #
        # Note that though you should be keeping track of the running         #
        # variance, you should normalize the data based on the standard       #
        # deviation (square root of variance) instead!                        # 
        # Referencing the original paper (https://arxiv.org/abs/1502.03167)   #
        # might prove to be helpful.                                          #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        
        # input x (N,D)
        N, D = x.shape
        
        # step1: calculate batch mean (D,)
        mu = np.mean(x, axis=0) 
        
        # step2: subtract mean vector of every trainings example
        xmu = x - mu
        
        
        #step3: following the lower branch - calculation denominator
        sq = xmu ** 2

        #step4: calculate batch variance (D,)
        var = 1.0/N * np.sum(sq, axis=0) # oppure var = np.var(x, axis=0)
        
        #step5: add eps for numerical stability, then sqrt
        sqrtvar = np.sqrt(var + eps)

        #step6: invert sqrtwar
        ivar = 1./sqrtvar

        #step7: execute normalization
        xhat = xmu * ivar

        #step8: Nor the two transformation steps
        gammax = gamma * xhat

        #step9
        out = gammax + beta

        # cache whats needed for the backward pass       
        cache = (xhat,gamma,xmu,ivar,sqrtvar,var,eps)
           
        
        """
        batch_variances = np.var(x, axis=0) #(D,)
        
        xhat = (x - batch_means)/np.sqrt(batch_variances + eps) # center to zero, normalize to one
        
        y = gamma * xhat + beta # Shift and scale
        
        out = y"""
        
        # Calculate the new running mean and variance
        running_mean = momentum * running_mean + (1 - momentum) * mu
        running_var = momentum * running_var + (1 - momentum) * var
        
        # Cache whats needed for the backward pass              
        """cache = (x, xhat, batch_means, batch_variances, eps, gamma, beta)"""

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == 'test':
        #######################################################################
        # TODO: Implement the test-time forward pass for batch normalization. #
        # Use the running mean and variance to normalize the incoming data,   #
        # then scale and shift the normalized data using gamma and beta.      #
        # Store the result in the out variable.                               #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #pass
        
        # The means and variances during test time will be the ones of the training time
        # calculated during the last iteration (NB this is the same as in torch7 but
        # not in the original paper
        x = (x - running_mean)/np.sqrt(running_var + eps)
        out = gamma * x + beta
        
        # Cache whats needed for the backward pass              
        # Do I need this ? I don't think so.. backward is not computed at test time..
        #cache = (xn, sample_variances, eps, gamma, beta)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var
    
    #print('inside forward batchnorm layer ', out.shape)

    return out, cache


def batchnorm_backward(dout, cache):
    """
    Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    # Referencing the original paper (https://arxiv.org/abs/1502.03167)       #
    # might prove to be helpful.                                              #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    # Unpack needed info from the forward pass
    xhat,gamma,xmu,ivar,sqrtvar,var,eps = cache
    """x, xhat, batch_means, batch_variances, eps, gamma, beta = cache"""
    
    # preliminary: get shape of layer
    N, D = dout.shape
    
    # step 9
    dgammax = dout
    dbeta = np.sum(dout, axis=0)    
    
    # step 8
    dgamma = np.sum(dgammax*xhat, axis=0)
    dxhat = gamma * dgammax
    
    # step 7
    dxmu1 = dxhat * ivar
    divar = np.sum(dxhat*xmu, axis = 0)
    
    # step 6
    dsqrtvar = divar * (-1.0) * np.power(sqrtvar, -2)
    
    # step 5
    dvar = 1.0/2 * dsqrtvar * np.power(var+eps, -1.0/2)
    
    # step 4 
    dsq = 1.0/N * np.ones((N, D)) * dvar
    
    # step 3
    dxmu2 = dsq * 2 * xmu # Not a matrix-matrix product but element-element multiplication ?
    
    # step 2
    dxmu = dxmu1 + dxmu2 
    dmu = -1.0*np.sum(dxmu,axis=0)
    dx1 = dxmu 
    
    # step 1
    dx2 = 1.0/N * np.ones((N, D)) * dmu
    
    dx = dx1 + dx2
    

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
    """
    Alternative backward pass for batch normalization.

    For this implementation you should work out the derivatives for the batch
    normalizaton backward pass on paper and simplify as much as possible. You
    should be able to derive a simple expression for the backward pass. 
    See the jupyter notebook for more hints.
     
    Note: This implementation should expect to receive the same cache variable
    as batchnorm_backward, but might not use all of the values in the cache.

    Inputs / outputs: Same as batchnorm_backward
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    #                                                                         #
    # After computing the gradient with respect to the centered inputs, you   #
    # should be able to compute gradients with respect to the inputs in a     #
    # single statement; our implementation fits on a single 80-character line.#
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    # Questo metodo e' non usando back propagation ma scrivendo esplicitamente su foglio le derivate - molto
    # piu' laborioso ma piu veloce rispetto a back propagation. 
    
    # Unpack needed info from the forward pass
    xhat,gamma,xmu,ivar,sqrtvar,var,eps = cache
    N, D = dout.shape
    
    # dbeta
    dbeta = np.sum(dout,axis=0)
    
    # dgamma
    dgamma = np.sum(dout*xhat, axis=0)
    
    # dx 
    dx = (1./N) * gamma * (var + eps)**(-1.0/2.0) * (N * dout - np.sum(dout, axis=0) - xmu * (var + eps)**(-1.0) * np.sum(dout*xmu, axis=0)) 
    
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def layernorm_forward(x, gamma, beta, ln_param):
    """
    Forward pass for layer normalization.

    During both training and test-time, the incoming data is normalized per data-point,
    before being scaled by gamma and beta parameters identical to that of batch normalization.
    
    Note that in contrast to batch normalization, the behavior during train and test-time for
    layer normalization are identical, and we do not need to keep track of running averages
    of any sort.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - ln_param: Dictionary with the following keys:
        - eps: Constant for numeric stability

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    out, cache = None, None
    eps = ln_param.get('eps', 1e-5)
    ###########################################################################
    # TODO: Implement the training-time forward pass for layer norm.          #
    # Normalize the incoming data, and scale and  shift the normalized data   #
    #  using gamma and beta.                                                  #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of  batch normalization, and inserting a line or two of  #
    # well-placed code. In particular, can you think of any matrix            #
    # transformations you could perform, that would enable you to copy over   #
    # the batch norm code and leave it almost unchanged?                      #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    """compact version
    layer_means = np.mean(x, axis=1).reshape(-1,1) # column vector of shape (N,)
    layer_vars = np.var(x, axis=1).reshape(-1,1) # column vector of shape (N,)
    xhat = (x-layer_means)/np.sqrt( layer_vars+eps ) #(N,D)
    out = gamma * xhat + beta
    """
    # Expanded version is more useful to cache intermediate variables
    # step1: calculate the array of data-point means (N,)
    mu = np.mean(x, axis=1)
    
    # step2: center every data-point
    xmu = x - mu.reshape(-1,1)
    
    # step3: following the lower branch - calculation denominator
    sq = xmu ** 2
    
    # step4: calculate the  array of data-point variances (N,)
    var = np.var(x, axis=1)
    
    # step5: add eps for numerical stability, then sqrt
    sqrtvar = np.sqrt(var + eps)
    
    # step6: invert sqrtwar
    ivar = 1./sqrtvar
    
    # step7: execute normalization
    xhat = xmu * ivar.reshape(-1,1)
    
    # step8: Nor the two transformation steps
    gammax = gamma * xhat

    # step9
    out = gammax + beta

    # cache whats needed for the backward pass
    # NB ivar è cachato come vettore riga
    cache = (xhat,gamma,xmu,ivar,sqrtvar,var,eps)
    
     
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def layernorm_backward(dout, cache):
    """
    Backward pass for layer normalization.

    For this implementation, you can heavily rely on the work you've done already
    for batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from layernorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for layer norm.                       #
    #                                                                         #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of batch normalization. The hints to the forward pass    #
    # still apply!                                                            #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    # Unpack needed info from the forward pass
    xhat,gamma,xmu,ivar,sqrtvar,var,eps = cache
    
    # preliminary: get shape of layer
    N, D = dout.shape
    
    # step 9
    dgammax = dout
    dbeta = np.sum(dout, axis=0)    
    
    # step 8
    dgamma = np.sum(dgammax*xhat, axis=0)
    dxhat = gamma * dgammax
    
    # here start the differences compared to batch normalization
    # step 7
    # dhat (N,D) ivar (N,)
    # Note that here I have to reshape ivar into a column vector to perform broadcast *
    dxmu1 = dxhat * ivar.reshape(-1, 1)
    divar = np.sum(dxhat*xmu, axis = 1)
    
    
    # step 6
    dsqrtvar = divar * (-1.0) * np.power(sqrtvar, -2)
    
   
    # step 5
    dvar = 1.0/2 * dsqrtvar * np.power(var+eps, -1.0/2)
    
    # step 4
    dsq = 1.0/D * np.ones((N, D)) * dvar.reshape(-1, 1)
    
    # step 3
    dxmu2 = dsq * 2 * xmu 
    
    # step 2
    dxmu = dxmu1 + dxmu2 
    dmu = -1.0*np.sum(dxmu,axis=1)
    dx1 = dxmu 
    
    # step 1
    dx2 = 1.0/D * np.ones((N, D)) * dmu.reshape(-1, 1)
    
    dx = dx1 + dx2
    

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
    """
    Performs the forward pass for (inverted) dropout.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We keep each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.

    NOTE: Please implement **inverted** dropout, not the vanilla version of dropout.
    See http://cs231n.github.io/neural-networks-2/#reg for more details.

    NOTE 2: Keep in mind that p is the probability of **keep** a neuron
    output; this might be contrary to some sources, where it is referred to
    as the probability of dropping a neuron output.
    """
    p, mode = dropout_param['p'], dropout_param['mode']
    if 'seed' in dropout_param:
        np.random.seed(dropout_param['seed'])

    mask = None
    out = None

    if mode == 'train':
        #######################################################################
        # TODO: Implement training phase forward pass for inverted dropout.   #
        # Store the dropout mask in the mask variable.                        #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        
        # p is probability of keeping a neuron
        
        # Create a random matrix with the same shape of x with uniform numbers between 0 and 1
        # Modify the mask into a boolean mask with True if the generated number is minor than p,
        # p being the probability of keeping a value.
        # In the 'inverted dropout' fashion, place a 1/p factor during training, thus
        # avoiding a *p during test time. 
        mask = (np.random.rand(*x.shape) < p)/p
        
        out = x * mask 

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == 'test':
        #######################################################################
        # TODO: Implement the test phase forward pass for inverted dropout.   #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # Given that at test time I cannot zero any neuron, which should be all active,
        # to preserve the identity I should multiply by p. However, given that during
        # training time I used an inverted Dropout by having put a factor (1/p), here I don't need anything. 
        out = x

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """
    Perform the backward pass for (inverted) dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param['mode']

    dx = None
    if mode == 'train':
        #######################################################################
        # TODO: Implement training phase backward pass for inverted dropout   #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        dx = dout * mask

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    elif mode == 'test':
        dx = dout 
    return dx


def conv_forward_naive(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width WW.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input. 
        

    During padding, 'pad' zeros should be placed symmetrically (i.e equally on both sides)
    along the height and width axes of the input. Be careful not to modfiy the original
    input x directly.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################
    
    N, C, H, W = x.shape
    F, C, HH, WW = w.shape
    S = conv_param['stride']
    pad = conv_param['pad']
    
    # Initialize to zero the out
    H_prime = int(1 + (H + 2 * pad - HH) / S)
    W_prime = int(1 + (W + 2 * pad - WW) / S)
    out = np.zeros( (N,F,H_prime,W_prime) )
    
    # x has shape (N, C, H, W) will result in shape (N,C, H+2p, W+2p)
    # npad is a tuple of (n_before, n_after) for each dimension
    npad = ((0, 0), (0,0), (pad, pad), (pad, pad))    
    xpadded = np.pad(x, pad_width=npad, mode='constant', constant_values=0)
    
    #print('input shape ', x.shape)
    #print('input padded shape ', xp.shape)
           
    
    for n in range(N):
        
        
        for f in range(F):
            
            filter_f = w[f,:,:,:] #(C, HH, WW)
            bias_f = b[f]         #(F,)
            
            #print('object number ', n, 'filter/bias number ', f)      
                  
            # Stride loops (horizontal and vertical axes)
            
            #print('filter shape',filter_f.shape)
            
            for j in range(W_prime):
                
                for i in range(H_prime):
                    
                    # i and j index the out object, while the indexes for the input object xp_n are
                    # in the row dimention: from i*S to i*S+HH
                    # in the column dimentions: from j*S to j*S+WW
                    #print('indexes', i*S, ':', i*S+HH, j*S, ':', j*S+WW)
                    
                    this_chunk = xpadded[n, :, i*S:i*S+HH, j*S:j*S+WW]
                    #print('chunk shape C, HH, WW', this_chunk.shape)
                    
                    # Compute multiplication and add bias
                    v = np.sum(this_chunk * filter_f) + b[f]
                                        
                    out[n,f,i,j] = v
                    #print('\n')
            
            
        
    #print('fine calcolo out shape ', out.shape)        
    

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    # unpack cache
    x,w,b,conv_param = cache
    pad = conv_param['pad']
    npad = ((0, 0), (0,0), (pad, pad), (pad, pad))    
    S = conv_param['stride']
    
    # Retrieving dimensions
    N, F, H_prime, W_prime = dout.shape
    F, C, HH, WW = w.shape
    N, C, H, W = x.shape
    
        
    # Initialize dx and dw
    dx = np.zeros_like(x)
    dw = np.zeros_like(w)
    
    # Initialize padded objects
    dx_padded = np.pad(dx, pad_width=npad, mode='constant', constant_values=0) 
    xpadded = np.pad(x, pad_width=npad, mode='constant', constant_values=0)
    
    
    # db
    # Sum dout over N, H', W'
    db = np.sum(dout, axis=(0,2,3))
    
    
    # dw (loop over dw dimentions)
    for f in range(F):    
        for c in range(C):
            for i in range(HH):      
                for j in range(WW):            

                    # per dare un certo valore (i,j) della matrice w la porzione di x e' questa qua sotto:
                    xpadded_chunk = xpadded[:, c, i:(i+H_prime)*S:S, j:(j+W_prime)*S:S]
                    #print('xpadded_chunk = ', xpadded_chunk.shape)
                    #print('dout[:, f, :, :] = ', dout[:, f, :, :].shape)
                    
                    dw[f, c, i, j] = np.sum(dout[:, f, :, :] * xpadded_chunk)

    
    # dx (this time we loop over the out dimentions)                
    for n in range(N):
        for f in range(F):
            for h_prime in range(H_prime):
                for w_prime in range(W_prime):                 
                        
                        dx_padded[n,:,h_prime*S:h_prime*S+HH, w_prime*S:w_prime*S+WW] += w[f,:,:,:] * dout[n,f,h_prime,w_prime]
    
    dx = dx_padded[:,:,pad:-pad,pad:-pad]

    

    
                
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """
    A naive implementation of the forward pass for a max-pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    No padding is necessary here. Output size is given by 

    Returns a tuple of:
    - out: Output data, of shape (N, C, H', W') where H' and W' are given by
      H' = 1 + (H - pool_height) / stride
      W' = 1 + (W - pool_width) / stride
    - cache: (x, pool_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the max-pooling forward pass                            #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    N, C, H, W = x.shape
    
    PH = pool_param['pool_height']
    PW = pool_param['pool_width']
    S = pool_param['stride']
    
    H_prime = int(1 + (H-PH)/S)
    W_prime = int(1 + (W-PW)/S)
    
    # Initialize out object, which will be of shape (N,C,H',W')
    out = np.zeros( (N,C,H_prime,W_prime) )
    
    # Loop over the dimentions of the out
    for n in range(N):
        for c in range(C):
            for h in range(H_prime):
                for w in range(W_prime):
                    
                    # Select the chunk of x to be pooled
                    chunk_to_pool = x[n,c, h*S:h*S+PH, w*S:w*S+PW]
                    
                    # Calculate the maximum value in this chunk
                    v = np.max(chunk_to_pool)
                    
                    out[n,c,h,w] = v
    

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a max-pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    ###########################################################################
    # TODO: Implement the max-pooling backward pass                           #
    ###########################################################################
    # Unpack cache
    x, pool_param = cache
    PH, PW, S = pool_param['pool_height'], pool_param['pool_width'], pool_param['stride']
    
    # Get maxpooling layer dimentions
    N, C, H_prime, W_prime = dout.shape
    
    # Initialize dx to zeros
    dx = np.zeros_like(x) # (N,C,H,W)

    # Loop over maxpooling layer dimentions
    for n in range(N):
        for c in range(C):
            for h in range(H_prime):
                for w in range(W_prime):
                    
                    # Select the chunk of x to be pooled (this is the same as the forward pass)
                    chunk_to_pool = x[n, c, h*S:h*S+PH, w*S:w*S+PW]
                    
                    # Calculate the maximum value in this chunk (this is the same as the forward pass)
                    v = np.max(chunk_to_pool)
                    
                    # Now get the indexes of the element which is the maximum of this chunk
                    # currently the whole dx is initialized to zeros, which is good (since the local gradient of the maxpooling layer is 0 for all elements but the maximum, which gives a 1.
                    max_row_index, max_col_index = np.where(chunk_to_pool == v)                    
                    max_row_index, max_col_index = int(max_row_index), int(max_col_index)
                    
                    # The element corresponding to the maximum becomes 1 (which is the
                    # local gradient of the maxpool layer) multyplied by the incoming gradient dout
                    dx[n, c, h*S+max_row_index, w*S+max_col_index] = 1.0 * dout[n,c,h,w]
                    
                    
    

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """
    Computes the forward pass for spatial batch normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance. momentum=0 means that
        old information is discarded completely at every time step, while
        momentum=1 means that new information is never incorporated. The
        default of momentum=0.9 should work well in most situations.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None

    ###########################################################################
    # TODO: Implement the forward pass for spatial batch normalization.       #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    # unpack just what I need
    N, C, H, W = x.shape
    
    # We can use batchnorm_forward function but that accepts a two-dimensional input of (N,D)
    # First transform x from (N, C, H, W) into (N, H, W, C) and then reshape into two-dimensional (N*H*W, C)
    x = x.transpose(0,2,3,1).reshape(-1, C)
    
    # Call vanilla batchnorm_forward
    # out is two-dimensional: (N*H*W, C)
    out, cache = batchnorm_forward(x, gamma, beta, bn_param)
    
    # First reshape (N*H*W, C) into (N,H,W,C) and then transpose to get (N,C,H,W)
    out = out.reshape(N,H,W,C).transpose(0,3,1,2)
    

           
    
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial batch normalization.      #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    # Unpack what I need
    N, C, H, W = dout.shape
    
    # Transform dout into shape accepted by batchnorm_backward, which is a two-dimnsional (N,D)
    # First transform dout from (N,C,H,W) into (N,H,W,C) and then into (N*H*W,C)
    dout = dout.transpose(0,2,3,1).reshape(-1,C)
    
    # dx will have the shape of type (N*H*W,C) while dgamma and dbeta of shape (C,)
    dx, dgamma, dbeta = batchnorm_backward(dout, cache)
    
    # we need to transform dx (N*H*W,C) into (N,C,H,W). 
    # dgamma and dbeta are already the correct shape
    dx = dx.reshape(N,H,W,C).transpose(0,3,1,2)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def spatial_groupnorm_forward(x, gamma, beta, G, gn_param):
    """
    Computes the forward pass for spatial group normalization.
    In contrast to layer normalization, group normalization splits each entry 
    in the data into G contiguous pieces, which it then normalizes independently.
    Per feature shifting and scaling are then applied to the data, in a manner identical to that of batch normalization and layer normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,) # maffe: actually it's shape is (1, C, 1, 1)
    - beta: Shift parameter, of shape (C,) # maffe: actually it's shape is (1, C, 1, 1)
    - G: Integer mumber of groups to split into, should be a divisor of C
    - gn_param: Dictionary with the following keys:
      - eps: Constant for numeric stability

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None
    eps = gn_param.get('eps',1e-5)
    ###########################################################################
    # TODO: Implement the forward pass for spatial group normalization.       #
    # This will be extremely similar to the layer norm implementation.        #
    # In particular, think about how you could transform the matrix so that   #
    # the bulk of the code is similar to both train-time batch normalization  #
    # and layer normalization!                                                # 
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    # Unpack what I need 
    N, C, H, W = x.shape
    
    """ Working version without using the NN layernorm_forward funcion explicitly"""
    """
    # Reshape x from (N,C,H,W) into (N, G, C//G, H, W)
    x = x.reshape(N, G, C//G, H, W)

    # Compute mean and std over the C/G, H and W axes
    # so for each N and G in x I'll have a mean and a std value.
    means = np.mean(x, axis=(2,3,4), keepdims=True)
    stds = np.std(x, axis=(2,3,4), keepdims=True)
    
    # Normalize x
    xhat = (x-means)/(stds+eps)   
    #print('xhat ', xhat.shape)
    
    # Reshape xhat from (N, G, C//G, H, W) to (N,C,H,W)
    xhat = xhat.reshape(N, C, H, W)
    
    #print('gamma ', gamma.shape)
    
    # xhat has shape (N,C,H,W) while gamma and beta (C,)
    out = xhat * gamma + beta"""
    
    # But the problem is that I have not cached anything for the backward pass. 
    
    
    """ Let's try to explicitly call the layernorm_forward func"""   
    # Change the shape of x to (N*H*W, G, C//G) 
    x = x.reshape(N, G, C//G, H, W) #(N, G, C//G, H, W)  
    x = x.transpose(0, 3,4, 1,2) #(N,H,W,G,C//G)
    x = x.reshape(N*H*W, G, C//G) #(N*H*W, G, C//G) 
    
    # Initialize the out with this shape
    out = np.zeros( (N*H*W, G, C//G) )
    
    # Change the shape of gamma and beta to (1, G, C//G) 
    gamma = gamma.reshape(1, G, C//G, 1, 1)
    gamma.transpose(0, 3,4, 1, 2)
    gamma = gamma.reshape(1, G, C//G)
    
    beta = beta.reshape(1, G, C//G, 1, 1)
    beta.transpose(0, 3,4, 1, 2)
    beta = beta.reshape(1, G, C//G)
    
    cache_from_forward = []
    
    for g in range(G):
        
        gamma_g = gamma[:,g,:] #(1, C//G) 
        beta_g = beta[:,g,:]  #(1, C//G) 
        x_g = x[:,g,:] #(N*H*W, C//G)
        
        #print('shape ', x_g.shape, gamma_g.shape, beta_g.shape)
        
        # out_g will have shape of x_g, which is (N*H*W, C//G)
        out_g, cache_g = layernorm_forward(x_g, gamma_g, beta_g, gn_param)
        
        # Fill the out tensor which has shape (N*H*W, G, C//G)
        out[:,g,:] = out_g
        
        # Cache from forward pass (g-th step)
        cache_from_forward.append(cache_g)
        
    # Reshape the out tensor to the original shape (N, C, H, W)
    out = out.reshape(N,H,W,G,C//G)
    out = out.transpose(0, 3,4, 1, 2)
    out = out.reshape(N, C, H, W)
    
    
    # Total cache from forward pass will be the cache from the layernorm_forward function for each g in G.
    # I also cache the G integer.
    cache = cache_from_forward, G
   

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def spatial_groupnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial group normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,) # maffe: actually it's shape is (1, C, 1, 1)
    - dbeta: Gradient with respect to shift parameter, of shape (C,) # maffe: actually it's shape is (1, C, 1, 1)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial group normalization.      #
    # This will be extremely similar to the layer norm implementation.        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    # Also here in the backward pass I'd like to make use of the layernorm_backward
    # function developed for NN. To do that, I'll have to reshape the objects into the shape
    # accepted by that function, which is in the form of dout=(N,D), dgamma=dbeta=(D,). In our case, the shape will
    # be for every g in G dout=(N*H*W, C//G), dgamma=dbeta=(C//G,). 
    # Note that dbeta could be easily calculated by simple dbeta = np.sum(dout, axis=(0,2,3))
    # but dx and dgamma will be much more problematic and I'll need to either calculate them by adapting the
    # computational graph to a CNN of to make use of the NN layernorm_backward. Let's chose this second road.
    
    # Unpack cache from forward
    cache_from_forward, G = cache
    N,C,H,W = dout.shape
       
    # Initialize objects to zero tensors
    dx = np.zeros( (N, C, H, W) )
    dgamma = np.zeros( (1, C, 1, 1) )
    dbeta = np.zeros( (1, C, 1, 1) )
    
    # Change the shapes to apply layernorm_backward function
    dout = dout.reshape(N, G, C//G, H, W) #(N, G, C//G, H, W)  
    dout = dout.transpose(0, 3,4, 1,2) #(N,H,W,G,C//G)
    dout = dout.reshape(N*H*W, G, C//G) #(N*H*W, G, C//G)
    
    dx = dx.reshape(N, G, C//G, H, W) #(N, G, C//G, H, W)  
    dx = dx.transpose(0, 3,4, 1,2) #(N,H,W,G,C//G)
    dx = dx.reshape(N*H*W, G, C//G) #(N*H*W, G, C//G)
    
    dgamma = dgamma.reshape(1, G, C//G, 1, 1)
    dgamma.transpose(0, 3,4, 1, 2)
    dgamma = dgamma.reshape(1, G, C//G)
    
    dbeta = dbeta.reshape(1, G, C//G, 1, 1)
    dbeta.transpose(0, 3,4, 1, 2)
    dbeta = dbeta.reshape(1, G, C//G)
    
    for g in range(G):
        
        # Get what I need
        dout_g = dout[:,g,:]
        cache_g = cache_from_forward[g]
        
        # Calculate backward gradients
        dx_g, dgamma_g, dbeta_g = layernorm_backward(dout_g, cache_g)
        
        # Assign gradients to relevant objects
        dx[:,g,:] = dx_g
        dgamma[:,g,:] = dgamma_g
        dbeta[:,g,:] = dbeta_g
        
       
    # Now reshape dx, dgamma and dbeta to original shapes
    dx = dx.reshape(N,H,W,G,C//G)
    dx = dx.transpose(0, 3,4, 1, 2)
    dx = dx.reshape(N, C, H, W)
    
    dgamma = dgamma.reshape(1, C, 1, 1)
    dbeta = dbeta.reshape(1, C, 1, 1)
    
    
    
    

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta


def svm_loss(x, y):
    """
    Computes the loss and gradient using for multiclass SVM classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    N = x.shape[0]
    correct_class_scores = x[np.arange(N), y]
    margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
    margins[np.arange(N), y] = 0
    loss = np.sum(margins) / N
    num_pos = np.sum(margins > 0, axis=1)
    dx = np.zeros_like(x)
    dx[margins > 0] = 1
    dx[np.arange(N), y] -= num_pos
    dx /= N
    return loss, dx


def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    shifted_logits = x - np.max(x, axis=1, keepdims=True)
    Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
    log_probs = shifted_logits - np.log(Z)
    probs = np.exp(log_probs)
    N = x.shape[0]
    loss = -np.sum(log_probs[np.arange(N), y]) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx
