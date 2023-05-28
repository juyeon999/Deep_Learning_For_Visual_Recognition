"""
Implements convolutional networks in PyTorch.
WARNING: you SHOULD NOT use ".to()" or ".cuda()" in each implementation block.
You are NOT allowed to use torch.nn ops, unless otherwise specified.
"""
import torch

from common.helpers import softmax_loss
from common import Solver
from fully_connected_networks import sgd_momentum, rmsprop, adam


def hello():
    """
    This is a sample function that we will try to import and run to ensure
    that our environment is correctly set up on Google Colab.
    """
    print('Hello from convolutional_networks.py!')


class Conv(object):

    @staticmethod
    def forward(x, w, b, conv_param):
        """
        A naive implementation of the forward pass for a convolutional layer.
        The input consists of N data points, each with C channels, height H and
        width W. We convolve each input with F different filters, where each
        filter spans all C channels and has height HH and width WW.

        Input:
        - x: Input data of shape (N, C, H, W)
        - w: Filter weights of shape (F, C, HH, WW)
        - b: Biases, of shape (F,)
        - conv_param: A dictionary with the following keys:
          - 'stride': The number of pixels between adjacent receptive fields
            in the horizontal and vertical directions.
          - 'pad': The number of pixels that is used to zero-pad the input.

        During padding, 'pad' zeros should be placed symmetrically (i.e equally
        on both sides) along the height and width axes of the input. Be careful
        not to modify the original input x directly.

        Returns a tuple of:
        - out: Output data of shape (N, F, H', W') where H' and W' are given by
          H' = 1 + (H + 2 * pad - HH) / stride
          W' = 1 + (W + 2 * pad - WW) / stride
        - cache: (x, w, b, conv_param)
        """
        out = None
        ######################################################################
        # TODO: Implement the convolutional forward pass.                    #
        # Hint: You can use function torch.nn.functional.pad for padding.    #
        # You are NOT allowed to use anything in torch.nn in other places.   #
        ######################################################################
        # Replace "pass" with your code (do not modify this line)
        stride = conv_param['stride']
        pad = conv_param['pad']

        N, C, H, W = x.shape
        F, _, HH, WW = w.shape        
        H1 = int((H - HH + 2 * pad)/ stride + 1)
        W1 = int((W - WW + 2 * pad)/ stride + 1)

        out = torch.zeros(N, F, H1, W1, dtype=x.dtype, device=x.device)  # output
        xp = torch.nn.functional.pad(x, (pad, pad, pad, pad))  # padding to first, last of the height and weight
        # Set the nested loop according to the dimension of output
        for f in range(F):
            for n in range(N):
                for h1 in range(H1):
                    for w1 in range(W1):
                        # Determine start point
                        # - Set the width/height start point to according to stride size
                        x_start = w1 * stride
                        y_start = h1 * stride
                        
                        # Determine the x range to convolution for
                        # - data point: for one data point
                        # - channel: every channel 
                        # - height: y_start to (y_start+HH); as kernel height
                        # - widht:  x_start to (x_start+WW); as kernel width
                        x2conv = xp[n, :, y_start:(y_start+HH), x_start:(x_start+WW)]
                        
                        # Dot product and summation over channel and add bias
                        out[n, f, h1, w1] = torch.sum(x2conv * w[f]) + b[f]
        
        # ref: https://github.com/AndreiKeino/EECS-498-007-598-005-Deep-Learning-for-Computer-Vision/blob/master/assignments/3/convolutional_networks_completed.ipynb
        ######################################################################
        #                          END OF YOUR CODE                          #
        ######################################################################
        cache = (x, w, b, conv_param)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        A naive implementation of the backward pass for a convolutional layer.
          Inputs:
        - dout: Upstream derivatives.
        - cache: A tuple of (x, w, b, conv_param) as in Conv.forward

        Returns a tuple of:
        - dx: Gradient with respect to x
        - dw: Gradient with respect to w
        - db: Gradient with respect to b
        """
        dx, dw, db = None, None, None
        ######################################################################
        # TODO: Implement the convolutional backward pass.                   #
        # Hint: You can use function torch.nn.functional.pad for padding.    #
        # You are NOT allowed to use anything in torch.nn in other places.   #
        ######################################################################
        # Replace "pass" with your code (do not modify this line)
        x, w, _, conv_param = cache
        stride = conv_param['stride']
        pad = conv_param['pad']
        N, C, H, W = x.shape
        F, _, HH, WW = w.shape
        _, _, hh, ww = dout.shape  # output size

        # Zero padding
        xp = torch.nn.functional.pad(x, (pad, pad, pad, pad))  # padding to first, last of the height and weight

        # Prepare for local gradients
        dw = torch.zeros_like(w)
        dxp = torch.zeros_like(xp)
        # Set the nested loop according to the dimension of dout     
        for f in range(F):
            for n in range(N):
                for hi in range(hh):
                    for wi in range(ww):
                        # Determine start point
                        # - Set the width/height start point to according to stride size
                        x_start = wi * stride
                        y_start = hi * stride

                        # Calculate local gradients of x_{n, c} and w_{f, c} for c = 1, ..., C
                        # - x_{n, c}: summation of w_{f, c} * full(dout_{n, f})
                        # - w_{f, c}: summation of x_{n, c} * dout_{n, c}
                        # - Performed for index other than c because the operation is being done for the entire c
                        # - for full operation, dout[ , , hi, wi]
                        dxp[n, :, y_start:(y_start + HH), x_start:(x_start + WW)] += w[f, :, :, :] * dout[n, f, hi, wi]
                        dw[f, :, :, :] += xp[n, :, y_start:(y_start + HH), x_start:(x_start + WW)] * dout[n, f, hi, wi]

        dx = dxp[:, :, pad:-pad, pad:-pad]  # exclude padded area
        db = torch.sum(dout, [0, 2, 3])  # summation over N, HH, WW

        ## ref: https://github.com/AndreiKeino/EECS-498-007-598-005-Deep-Learning-for-Computer-Vision/blob/master/assignments/3/convolutional_networks_completed.ipynb
        ######################################################################
        #                          END OF YOUR CODE                          #
        ######################################################################
        return dx, dw, db


class MaxPool(object):

    @staticmethod
    def forward(x, pool_param):
        """
        A naive implementation of the forward pass for a max-pooling layer.

        Inputs:
        - x: Input data, of shape (N, C, H, W)
        - pool_param: dictionary with the following keys:
          - 'pool_height': The height of each pooling region
          - 'pool_width': The width of each pooling region
          - 'stride': The distance between adjacent pooling regions
        No padding is necessary here.

        Returns a tuple of:
        - out: Output of shape (N, C, H', W') where H' and W' are given by
          H' = 1 + (H - pool_height) / stride
          W' = 1 + (W - pool_width) / stride
        - cache: (x, pool_param)
        """
        out = None
        ######################################################################
        # TODO: Implement the max-pooling forward pass.                      #
        ######################################################################
        # Replace "pass" with your code (do not modify this line)
        pool_width = pool_param['pool_width']
        pool_height = pool_param['pool_height']
        stride = pool_param['stride']

        N, F, H, W = x.shape
        H1 = int(1 + (H - pool_height) / stride)
        W1 = int(1 + (W - pool_width) / stride)

        out = torch.zeros(N, F, H1, W1, dtype=x.dtype, device=x.device)  # N and F are equivalent to x

        for f in range(F):
            for n in range(N):
                for hi in range(H1):
                    for wi in range(W1):
                        # Determine start point
                        x_start = wi * stride
                        y_start = hi * stride

                        # Specify the scope to which pooling is applied
                        scope = x[n, f, y_start:(y_start+pool_height), x_start:(x_start+pool_width)]
                        out[n, f, hi, wi] = torch.max(scope)

        ## No reference (I wrote it myself by referring to the Conv class code above)
        ######################################################################
        #                          END OF YOUR CODE                          #
        ######################################################################
        cache = (x, pool_param)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        A naive implementation of the backward pass for a max-pooling layer.
        Inputs:
        - dout: Upstream derivatives
        - cache: A tuple of (x, pool_param) as in the forward pass.
        Returns:
        - dx: Gradient with respect to x
        """
        dx = None
        ######################################################################
        # TODO: Implement the max-pooling backward pass.                     #
        ######################################################################
        # Replace "pass" with your code (do not modify this line)
        x, pool_param = cache
        pool_width = pool_param['pool_width']
        pool_height = pool_param['pool_height']
        stride = pool_param['stride']

        N, F, H, W = x.shape
        _, _, H1, W1 = dout.shape

        # Make zero filled tensor for gradient matrix of matrix x
        dx = torch.zeros(N, F, H, W, dtype=x.dtype, device=x.device)

        for f in range(F):
            for n in range(N):
                for hi in range(H1):
                    for wi in range(W1):
                        # Determine start point
                        x_start = wi * stride
                        y_start = hi * stride

                        # Get the local gradient (per max pool scope)
                        # - maximum element: local gradient is alive
                        # - non-maximum element: local gradient is dead 
                        scope = x[n, f, y_start:(y_start+pool_height), x_start:(x_start+pool_width)]
                        i_y = torch.argmax(scope) // pool_width  # height of maximum element of scope
                        i_x = torch.argmax(scope) % pool_width   # width of maximum element of scope

                        # Assign local gradient
                        # - upstream gradient: (hi, wi)-th element of dout
                        # - local gradient: all zero except (y_start+i_y, x_start+i_x)-th element
                        dx[n, f, (y_start+i_y), (x_start+i_x)] = dout[n, f, hi, wi]  

        ## No reference (I wrote it myself by referring to the Conv class code above)
        ######################################################################
        #                          END OF YOUR CODE                          #
        ######################################################################
        return dx


class ThreeLayerConvNet(object):
    """
    A three-layer convolutional network with the following architecture:
    conv - relu - 2x2 max pool - linear - relu - linear - softmax
    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self,
                 input_dims=(3, 32, 32),
                 num_filters=32,
                 filter_size=7,
                 hidden_dim=100,
                 num_classes=10,
                 weight_scale=1e-3,
                 reg=0.0,
                 dtype=torch.float,
                 device='cpu'):
        """
        Initialize a new network.
        Inputs:
        - input_dims: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Width/height of filters to use in convolutional layer
        - hidden_dim: Number of units to use in fully-connected hidden layer
        - num_classes: Number of scores to produce from the final linear layer.
        - weight_scale: Scalar giving standard deviation for random
          initialization of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: A torch data type object; all computations will be performed
          using this datatype. float is faster but less accurate, so you
          should use double for numeric gradient checking.
        - device: device to use for computation. 'cpu' or 'cuda'
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype

        ######################################################################
        # TODO: Initialize weights and biases for three-layer convolutional  #
        # network. Weights should be initialized from the Gaussian           #
        # distribution with the mean of 0.0 and the standard deviation of    #
        # weight_scale; biases should be initialized to zero. All weights    #
        # and biases should be stored in the dictionary self.params.         #
        # Store weights and biases for the convolutional layer using the     #
        # keys 'W1' and 'b1'; use keys 'W2' and 'b2' for the weights and     #
        # biases of the hidden linear layer, and keys 'W3' and 'b3' for the  #
        # weights and biases of the output linear layer.                     #
        #                                                                    #
        # IMPORTANT: For this assignment, you can assume that the padding    #
        # and stride of the first convolutional layer are chosen so that     #
        # **the width and height of the input are preserved**. Take a        #
        # look at the start of the loss() function to see how that happens.  #
        ######################################################################
        # Replace "pass" with your code (do not modify this line)
        C, H, W = input_dims
        pool_height, pool_width = 2, 2
        stride = 2  # for preserve input width and height
        H1 = 1 + (H - pool_height) / stride  # height after max pooling
        W1 = 1 + (W - pool_width) / stride   # width after max pooling

        # Weights and biases for the convolutional layer
        self.params['W1'] = weight_scale * \
            torch.rand(size=(num_filters, C, filter_size, filter_size), dtype=self.dtype, device=device)
        self.params['b1'] = torch.zeros(num_filters, dtype=self.dtype, device=device)
        
        # The weights and biases of the hidden linear layer
        flatten_size = int(num_filters * H1 * W1)
        self.params['W2'] = weight_scale * \
            torch.rand(size=(flatten_size, hidden_dim), dtype=self.dtype, device=device)
        self.params['b2'] = torch.zeros(hidden_dim, dtype=self.dtype, device=device)
        
        # The weights and biases of the output linear layer
        self.params['W3'] = weight_scale * \
            torch.rand(size=(hidden_dim, num_classes), dtype=self.dtype, device=device)
        self.params['b3'] = torch.zeros(num_classes, dtype=self.dtype, device=device)

        ## No reference (I wrote it myself by referring to the Conv class code above)
        ######################################################################
        #                          END OF YOUR CODE                          #
        ######################################################################

    def save(self, path):
        checkpoint = {
          'reg': self.reg,
          'dtype': self.dtype,
          'params': self.params,
        }
        torch.save(checkpoint, path)
        print("Saved in {}".format(path))

    def load(self, path):
        checkpoint = torch.load(path, map_location='cpu')
        self.params = checkpoint['params']
        self.dtype = checkpoint['dtype']
        self.reg = checkpoint['reg']
        print("load checkpoint file: {}".format(path))

    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.
        Input / output: Same API as TwoLayerNet.
        """
        X = X.to(self.dtype)
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']

        # Pass conv_param to the forward pass for the convolutional layer.
        # Padding and stride chosen to preserve the input spatial size.
        filter_size = W1.shape[2]
        conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}

        # Pass pool_param to the forward pass for the max-pooling layer.
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        scores = None
        ######################################################################
        # TODO: Implement the forward pass for three-layer convolutional     #
        # net, computing the class scores for X and storing them in the      #
        # scores variable.                                                   #
        # Use sandwich layers if Linear or Conv layers followed by ReLU      #
        # and/or Pool layers for efficient implementation.                   #
        ######################################################################
        # Replace "pass" with your code (do not modify this line)
        # conv - relu - 2x2 max pool - linear - relu - linear - 1
        out_conv, cache_conv = Conv_ReLU_Pool.forward(X, W1, b1, conv_param, pool_param)
        out_hidden, cache_hidden= Linear_ReLU.forward(out_conv, W2, b2)
        scores, cache_output = Linear.forward(out_hidden, W3, b3)       

        ## ref: No reference (My own code)
        ######################################################################
        #                          END OF YOUR CODE                          #
        ######################################################################

        if y is None:
            return scores

        loss, grads = 0.0, {}
        ######################################################################
        # TODO: Implement the backward pass for three-layer convolutional    #
        # net, storing the loss and gradients in the loss and grads.         #
        # Compute the data loss using softmax, and make sure that grads[k]   #
        # holds the gradients for self.params[k]. Don't forget to add        #
        # L2 regularization!                                                 #
        # NOTE: To ensure your implementation matches ours and you pass the  #
        # automated tests, make sure that your L2 regularization includes    #
        # a factor of 0.5 to simplify the expression for the gradient.       #
        ######################################################################
        # Replace "pass" with your code (do not modify this line)
        # Loss
        loss, dout = softmax_loss(scores, y)
        loss += 0.5 * self.reg * (torch.sum(W1 ** 2) + torch.sum(W2 ** 2) + torch.sum(W3 ** 2))
        
        # Grads
        dx_out, dw, db = Linear.backward(dout, cache_output)
        grads['W3'] = dw + self.reg * W3
        grads['b3'] = db
        
        dx_hidden, dw, db = Linear_ReLU.backward(dx_out, cache_hidden)
        grads['W2'] = dw + self.reg * W2
        grads['b2'] = db

        dx_conv, dw, db = Conv_ReLU_Pool.backward(dx_hidden, cache_conv)
        # grads['W1'] = dw
        grads['W1'] = dw + self.reg * W1
        grads['b1'] = db    

        ## ref: No reference (My own code)
        ######################################################################
        #                          END OF YOUR CODE                          #
        ######################################################################

        return loss, grads


class DeepConvNet(object):
    """
    A convolutional neural network with an arbitrary number of convolutional
    layers in VGG-Net style. All convolution layers will use kernel size 3 and
    padding 1 to preserve the feature map size, and all pooling layers will be
    max pooling layers with 2x2 receptive fields and a stride of 2 to halve the
    size of the feature map.

    The network will have the following architecture:

    {conv - [batchnorm?] - relu - [pool?]} x (L - 1) - linear

    Each {...} structure is a "macro layer" consisting of a convolution layer,
    an optional batch normalization layer, a ReLU nonlinearity, and an optional
    pooling layer. After L-1 such macro layers, a single fully-connected layer
    is used to predict the class scores.

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """
    def __init__(self,
                 input_dims=(3, 32, 32),
                 num_filters=[8, 8, 8, 8, 8],
                 max_pools=[0, 1, 2, 3, 4],
                 batchnorm=False,
                 num_classes=10,
                 weight_scale=1e-3,
                 reg=0.0,
                 dtype=torch.float,
                 device='cpu'):
        """
        Initialize a new network.

        Inputs:
        - input_dims: Tuple (C, H, W) giving size of input data
        - num_filters: List of length (L - 1) giving the number of
          convolutional filters to use in each macro layer.
        - max_pools: List of integers giving the indices of the macro
          layers that should have max pooling (zero-indexed).
        - batchnorm: Whether to include batch normalization in each macro layer
        - num_classes: Number of scores to produce from the final linear layer.
        - weight_scale: Scalar giving standard deviation for random
          initialization of weights, or the string 'kaiming' to use Kaiming
          initialization instead
        - reg: Scalar giving L2 regularization strength. L2 regularization
          should only be applied to convolutional and fully-connected weight
          matrices; it should not be applied to biases or to batchnorm scale
          and shifts.
        - dtype: A torch data type object; all computations will be performed
          using this dtype. float is faster but less accurate, so you should
          use double for numeric gradient checking.
        - device: device to use for computation. 'cpu' or 'cuda'
        """
        self.params = {}
        self.num_layers = len(num_filters)+1
        self.max_pools = max_pools
        self.batchnorm = batchnorm
        self.reg = reg
        self.dtype = dtype

        if device == 'cuda':
            device = 'cuda:0'

        ######################################################################
        # TODO: Initialize the parameters for the DeepConvNet. All weights,  #
        # biases, and batchnorm scale and shift parameters should be stored  #
        # in the dictionary self.params, where the keys should be in the     #
        # form of 'W#', 'b#', 'gamma#', and 'beta#' with 1-based indexing.   #
        # Weights for Conv and Linear layers should be initialized from the  #
        # Gaussian distribution with the mean of 0.0 and the standard        #
        # deviation of weight_scale; however, if weight_scale == 'kaiming',  #
        # then you should call kaiming_initializer instead. Biases should be #
        # initialized to zeros. Batchnorm scale (gamma) and shift (beta)     #
        # parameters should be initialized to ones and zeros, respectively.  #
        ######################################################################
        # Replace "pass" with your code (do not modify this line)   
        C, H, W = input_dims
        K = 3  # pad 1, stride 1; preserve input size
        pool_K = 2  # 2 by 2 receptive field
        pool_stride = 2
        H_conv_out, W_conv_out = H, W  # Set for non pooling aritecture

        # 1. For Conv layer
        for i, F in enumerate(num_filters):
            # For Wi and bi
            if weight_scale == 'kaiming':
                self.params['W' + str(i+1)] = kaiming_initializer(Din=C, Dout=F, K=K, dtype=self.dtype, device=device)
            else:
                self.params['W' + str(i+1)] = weight_scale * \
                    torch.rand(size=(F, C, K, K), dtype=self.dtype, device=device)
            self.params['b' + str(i+1)] = torch.zeros(F, dtype=self.dtype, device=device)  # bias size is depend on F
            
            # For BatchNorm
            # mean vector and std per channel => size is (F, )
            if self.batchnorm:
                self.params['gamma' + str(i+1)] = torch.ones(F, dtype=self.dtype, device=device)
                self.params['beta' + str(i+1)] = torch.zeros(F, dtype=self.dtype, device=device)
            
            # For output size
            # - If pooling layer exist, update the height and width of output
            if i in self.max_pools:
                H_conv_out = int(1 + (H_conv_out - pool_K) / pool_stride)
                W_conv_out = int(1 + (W_conv_out - pool_K) / pool_stride)
            
            C = F  # filter num of this conv layer will be the channel of next conv layer
             
        # 2. For Linear layer
        flatten_size = F * H_conv_out * W_conv_out  # flatten to linear layer
        if weight_scale == 'kaiming':
            self.params['W' + str(self.num_layers)] = kaiming_initializer(Din=flatten_size, Dout=num_classes, K=None, dtype=self.dtype, device=device)
        else:
            self.params['W' + str(self.num_layers)] = weight_scale * \
                torch.rand(size=(flatten_size, num_classes), dtype=self.dtype, device=device)
        self.params['b' + str(self.num_layers)] = torch.zeros(num_classes, dtype=self.dtype, device=device)

        # ref: https://github.com/AndreiKeino/EECS-498-007-598-005-Deep-Learning-for-Computer-Vision/blob/master/assignments/3/convolutional_networks_completed.ipynb
        ######################################################################
        #                          END OF YOUR CODE                          #
        ######################################################################

        # With batch normalization we need to keep track of running
        # means and variances, so we need to pass a special bn_param
        # object to each batch normalization layer. You should pass
        # self.bn_params[0] to the forward pass of the first batch
        # normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.batchnorm:
            self.bn_params = [{'mode': 'train'}
                              for _ in range(len(num_filters))]

        # Check that we got the right number of parameters
        if not self.batchnorm:
            params_per_macro_layer = 2  # weight and bias
        else:
            params_per_macro_layer = 4  # weight, bias, scale, shift
        num_params = params_per_macro_layer * len(num_filters) + 2
        msg = 'self.params has the wrong number of ' \
              'elements. Got %d; expected %d'
        msg = msg % (len(self.params), num_params)
        assert len(self.params) == num_params, msg

        # Check that all parameters have the correct device and dtype:
        for k, param in self.params.items():
            msg = 'param "%s" has device %r; should be %r' \
                  % (k, param.device, device)
            assert param.device == torch.device(device), msg
            msg = 'param "%s" has dtype %r; should be %r' \
                  % (k, param.dtype, dtype)
            assert param.dtype == dtype, msg

    def save(self, path):
        checkpoint = {
          'reg': self.reg,
          'dtype': self.dtype,
          'params': self.params,
          'num_layers': self.num_layers,
          'max_pools': self.max_pools,
          'batchnorm': self.batchnorm,
          'bn_params': self.bn_params,
        }
        torch.save(checkpoint, path)
        print("Saved in {}".format(path))

    def load(self, path, dtype, device):
        checkpoint = torch.load(path, map_location='cpu')
        self.params = checkpoint['params']
        self.dtype = dtype
        self.reg = checkpoint['reg']
        self.num_layers = checkpoint['num_layers']
        self.max_pools = checkpoint['max_pools']
        self.batchnorm = checkpoint['batchnorm']
        self.bn_params = checkpoint['bn_params']

        for p in self.params:
            self.params[p] = self.params[p].type(dtype).to(device)

        for i in range(len(self.bn_params)):
            for p in ["running_mean", "running_var"]:
                self.bn_params[i][p] = \
                    self.bn_params[i][p].type(dtype).to(device)

        print("load checkpoint file: {}".format(path))

    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the deep convolutional
        network.
        Input / output: Same API as ThreeLayerConvNet.
        """
        X = X.to(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params since they
        # behave differently during training and testing.
        if self.batchnorm:
            for bn_param in self.bn_params:
                bn_param['mode'] = mode
        scores = None

        # pass conv_param to the forward pass for the
        # convolutional layer
        # Padding and stride chosen to preserve the input
        # spatial size
        filter_size = 3
        conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        scores = None
        ######################################################################
        # TODO: Implement the forward pass for DeepConvNet, computing the    #
        # class scores for X and storing them in the scores variable.        #
        # Use sandwich layers if Linear or Conv layers followed by ReLU      #
        # and/or Pool layers for efficient implementation.                   #
        ######################################################################
        # Replace "pass" with your code (do not modify this line)
        cache_list = []
        X_in = X.clone()
        for i in range(self.num_layers-1):
            W = self.params['W' + str(i+1)]
            b = self.params['b' + str(i+1)]
            if self.batchnorm:
                gamma =self.params['gamma' + str(i+1)]
                beta =self.params['beta' + str(i+1)]
                if i in self.max_pools:
                    # bn + pool
                    X_in, cache = Conv_BatchNorm_ReLU_Pool.forward(X_in, W, b, gamma, beta, conv_param, bn_param, pool_param)
                    cache_list.append(cache)
                else:
                    # bn + not pool
                    X_in, cache = Conv_BatchNorm_ReLU.forward(X_in, W, b, gamma, beta, conv_param, bn_param)
                    cache_list.append(cache)
            else:
                if i in self.max_pools:
                    # not bn + pool
                    X_in, cache = Conv_ReLU_Pool.forward(X_in, W, b, conv_param, pool_param)
                    cache_list.append(cache)
                else:
                    # not bn + not pool
                    X_in, cache = Conv_ReLU.forward(X_in, W, b, conv_param)
                    cache_list.append(cache)

        W = self.params['W' + str(self.num_layers)]
        b = self.params['b' + str(self.num_layers)]
        scores, cache = Linear.forward(X_in, W, b)
        cache_list.append(cache)

        # ref: No reference(My own code)
        ######################################################################
        #                          END OF YOUR CODE                          #
        ######################################################################

        if y is None:
            return scores

        loss, grads = 0, {}
        ######################################################################
        # TODO: Implement the backward pass for the DeepConvNet, storing the #
        # loss and gradients in the loss and grads variables.                #
        # Compute the data loss using softmax, and make sure that grads[k]   #
        # holds the gradients for self.params[k]. Don't forget to add        #
        # L2 regularization!                                                 #
        # NOTE: To ensure your implementation matches ours and you pass the  #
        # automated tests, make sure that your L2 regularization includes    #
        # a factor of 0.5 to simplify the expression for the gradient.       #
        ######################################################################
        # Replace "pass" with your code (do not modify this line)
        # 1. Loss
        loss, dout = softmax_loss(scores, y)
        for i in range(self.num_layers-1):
            W = self.params['W' + str(i+1)]
            b = self.params['b' + str(i+1)]
            loss += 0.5 * self.reg * torch.sum(W ** 2)  # Add regularization

        # 2-1. Grads for Linear layer
        W = self.params['W' + str(self.num_layers)]
        dx, dw, db = Linear.backward(dout, cache_list[self.num_layers-1])
        grads['W' + str(self.num_layers)] = dw + self.reg * W
        grads['b' + str(self.num_layers)] = db
        
        # 2-1. Grads for Conv layer
        for i in reversed(range(self.num_layers-1)):     
            W = self.params['W' + str(i+1)]
            b = self.params['b' + str(i+1)]   

            if self.batchnorm:
                gamma =self.params['gamma' + str(i+1)]
                beta =self.params['beta' + str(i+1)]
                if i in self.max_pools:
                    # bn + pool
                    dx, dw, db, dgamma, dbeta = Conv_BatchNorm_ReLU_Pool.backward(dx, cache_list[i])
                    cache_list.append(cache)
                else:
                    # bn + not pool
                    dx, dw, db, dgamma, dbeta = Conv_BatchNorm_ReLU.backward(dx, cache_list[i])
                    cache_list.append(cache)
                grads['gamma' + str(i+1)] = dgamma
                grads['beta' + str(i+1)] = dbeta
            else:
                if i in self.max_pools:
                    # not bn + pool
                    dx, dw, db = Conv_ReLU_Pool.backward(dx, cache_list[i])
                    cache_list.append(cache)
                else:
                    # not bn + not pool
                    dx, dw, db = Conv_ReLU.backward(dx, cache_list[i])
                    cache_list.append(cache)
            grads['W' + str(i+1)] = dw + self.reg * W
            grads['b' + str(i+1)] = db

        ## ref: No reference (My own code)
        ######################################################################
        #                          END OF YOUR CODE                          #
        ######################################################################

        return loss, grads


def find_overfit_parameters():
    weight_scale = 2e-3   # Experiment with this!
    learning_rate = 1e-5  # Experiment with this!
    ##########################################################################
    # TODO: Change weight_scale and learning_rate so your model achieves     #
    # 100% training accuracy within 30 epochs.                               #
    ##########################################################################
    # Replace "pass" with your code (do not modify this line)
    weight_scale = 5e-2
    learning_rate = 1e-4

    # ref: https://github.com/AndreiKeino/EECS-498-007-598-005-Deep-Learning-for-Computer-Vision/blob/master/assignments/3/convolutional_networks_completed.ipynb
    ##########################################################################
    #                            END OF YOUR CODE                            #
    ##########################################################################
    return weight_scale, learning_rate


def kaiming_initializer(Din, Dout, K=None, relu=True, device='cpu',
                        dtype=torch.float32):
    """
    Implement Kaiming initialization for linear and convolution layers.

    Inputs:
    - Din, Dout: Integers giving the number of input and output dimensions
      for this layer
    - K: If K is None, then initialize weights for a linear layer with
      Din input dimensions and Dout output dimensions. Otherwise if K is
      a nonnegative integer then initialize the weights for a convolution
      layer with Din input channels, Dout output channels, and a kernel size
      of KxK.
    - relu: If ReLU=True, then initialize weights with a gain of 2 to
      account for a ReLU nonlinearity (Kaiming initialization); otherwise
      initialize weights with a gain of 1 (Xavier initialization).
    - device, dtype: The device and datatype for the output tensor.

    Returns:
    - weight: A torch Tensor giving initialized weights for this layer.
      For a linear layer it should have shape (Din, Dout); for a
      convolution layer it should have shape (Dout, Din, K, K).
    """
    gain = 2. if relu else 1.
    weight = None
    if K is None:
        ######################################################################
        # TODO: Implement the Kaiming initialization for linear layer.       #
        # The weight_scale is sqrt(gain / fan_in), where gain is 2 if ReLU   #
        # is followed by the layer, or 1 if not, and fan_in = Din.           #
        # The output should be a tensor in the designated size, dtype,       #
        # and device.                                                        #
        ######################################################################
        # Replace "pass" with your code (do not modify this line)

        # For linear layer, weight shape is (Din, Dout)
        weight_scale = torch.tensor(gain / Din, dtype=torch.float32, device='cpu')
        weight = torch.sqrt(weight_scale) * torch.randn(Din, Dout, dtype = dtype, device=device)
        
        ## ref: https://github.com/AndreiKeino/EECS-498-007-598-005-Deep-Learning-for-Computer-Vision/blob/master/assignments/3/convolutional_networks_completed.ipynb
        ######################################################################
        #                          END OF YOUR CODE                          #
        ######################################################################
    else:
        ######################################################################
        # TODO: Implement Kaiming initialization for convolutional layer.    #
        # The weight_scale is sqrt(gain / fan_in), where gain is 2 if ReLU   #
        # is followed by the layer, or 1 if not, and fan_in = Din * K * K.   #
        # The output should be a tensor in the designated size, dtype,       #
        # and device.                                                        #
        ######################################################################
        # Replace "pass" with your code (do not modify this line)

        # For Conv layer, weight shape is (Din, Dout, K, K)
        weight_scale = torch.tensor(gain / (Din * K * K), dtype=torch.float32, device='cpu')
        weight = torch.sqrt(weight_scale) * torch.randn(Dout, Din, K, K, dtype = dtype, device=device)

        ## ref: https://github.com/AndreiKeino/EECS-498-007-598-005-Deep-Learning-for-Computer-Vision/blob/master/assignments/3/convolutional_networks_completed.ipynb
        ######################################################################
        #                          END OF YOUR CODE                          #
        ######################################################################
    return weight


def create_convolutional_solver_instance(data_dict, dtype, device):
    model = None
    solver = None
    ##########################################################################
    # TODO: Train the best DeepConvNet on CIFAR-10 within 60 seconds.        #
    # Hint: You can use any optimizer you implemented in                     #
    # fully_connected_networks.py, which we imported for you.                #
    ##########################################################################
    # Replace "pass" with your code (do not modify this line)
    input_dims = data_dict['X_train'].shape[1:]
    weight_scale =  'kaiming'
    reg =  0. 
    num_epochs = 10
    batch_size = 128
    learning_rate =  9.00E-04
    
    model = DeepConvNet(input_dims=input_dims, num_classes=10,
                        num_filters=([8] * 3) + ([32] * 3) + ([128] * 3),
                        max_pools=[3, 6, 9],
                        weight_scale=weight_scale,
                        reg=reg,
                        dtype=torch.float32,
                        device='cuda'
                        )

    solver = Solver(model, data_dict,
                    num_epochs=num_epochs, batch_size=batch_size,
                    update_rule=adam,
                    optim_config={
                        'learning_rate': learning_rate,
                    },
                    print_every=20, device='cuda')

    # ref: https://github.com/AndreiKeino/EECS-498-007-598-005-Deep-Learning-for-Computer-Vision/blob/master/assignments/3/convolutional_networks_completed.ipynb
    ##########################################################################
    #                            END OF YOUR CODE                            #
    ##########################################################################
    return solver


class BatchNorm(object):

    @staticmethod
    def forward(x, gamma, beta, bn_param):
        """
        Forward pass for batch normalization.

        During training the sample mean and (uncorrected) sample variance
        are computed from minibatch statistics and used to normalize the
        incoming data. During training we also keep an exponentially decaying
        running mean of the mean and variance of each feature, and these
        averages are used to normalize data at test-time.

        At each time step, we update the running averages for mean and
        variance using an exponential decay based on the momentum parameter:

        running_mean = momentum * running_mean + (1 - momentum) * sample_mean
        running_var = momentum * running_var + (1 - momentum) * sample_var

        Note that the batch normalization paper suggests a different
        test-time behavior: they compute sample mean and variance for
        each feature using a large number of training images rather than
        using a running average. For this implementation we have chosen to use
        running averages instead since they do not require an additional
        estimation step; the PyTorch implementation of batch normalization
        also uses running averages.

        Input:
        - x: Data of shape (N, D)
        - gamma: Scale parameter of shape (D,)
        - beta: Shift parameter of shape (D,)
        - bn_param: Dictionary with the following keys:
          - mode: 'train' or 'test'; required
          - eps: Constant for numeric stability
          - momentum: Constant for running mean / variance.
          - running_mean: Array of shape (D,) giving running mean
            of features
          - running_var Array of shape (D,) giving running variance
            of features

        Returns a tuple of:
        - out: of shape (N, D)
        - cache: A tuple of values needed in the backward pass
        """
        mode = bn_param['mode']
        eps = bn_param.get('eps', 1e-5)
        momentum = bn_param.get('momentum', 0.9)

        N, D = x.shape
        running_mean = \
            bn_param.get('running_mean',
                         torch.zeros(D, dtype=x.dtype, device=x.device))
        running_var = \
            bn_param.get('running_var',
                         torch.ones(D, dtype=x.dtype, device=x.device))

        out, cache = None, None
        if mode == 'train':
            ##################################################################
            # TODO: Implement the training-time forward pass for batchnorm.  #
            # Use minibatch statistics to compute the mean and variance.     #
            # Use the mean and variance to normalize the incoming data, and  #
            # then scale and shift the normalized data using gamma and beta. #
            #                                                                #
            # You should store the output in the variable out.               #
            # Any intermediates that you need for the backward pass should   #
            # be stored in the cache variable.                               #
            #                                                                #
            # You should also use your computed sample mean and variance     #
            # together with the momentum variable to update the running mean #
            # and running variance, storing your result in the running_mean  #
            # and running_var variables.                                     #
            #                                                                #
            # Note that though you should be keeping track of the running    #
            # variance, you should normalize the data based on the standard  #
            # deviation (square root of variance) instead!                   #
            # Referencing the original paper                                 #
            # (https://arxiv.org/abs/1502.03167) might prove to be helpful.  #
            ##################################################################
            # Replace "pass" with your code (do not modify this line)
            
            # Calculate sample mean, variance, sd per channel
            mu = torch.sum(x, dim=0) / N  # sample mean; summation over channel and divided by N
            subtraction = x - mu
            var = torch.sum((subtraction) ** 2, axis=0) / (N - 1)  # sample variance; summation about difference over channel and divided by (N-1) 
            sd = torch.sqrt(var + eps)  # add small constant for numerical stability and squared of it
            sd_inv = 1. / sd

            # Normalization
            x_hat = subtraction / sd
            
            # Scale and Shift the normalized data using gamma and beta
            out = gamma * x_hat + beta

            # Store variables in cache
            cache = (x_hat, gamma, subtraction, sd_inv, sd, var, eps)

            # Moving average
            running_mean = momentum * running_mean + (1-momentum) * mu
            running_var = momentum * running_var + (1-momentum) * var

            ## ref: https://github.com/AndreiKeino/EECS-498-007-598-005-Deep-Learning-for-Computer-Vision/blob/master/assignments/3/convolutional_networks_completed.ipynb
            ##################################################################
            #                        END OF YOUR CODE                        #
            ##################################################################
        elif mode == 'test':
            ##################################################################
            # TODO: Implement the test-time forward pass for batchnorm.      #
            # Use the running mean and variance to normalize the incoming    #
            # data, and then scale and shift the normalized data using gamma #
            # and beta. Store the result in the out variable.                #
            ##################################################################
            # Replace "pass" with your code (do not modify this line)

            # In test time, use running mean and running variance to Normalized
            # and usd gamma and beta to scale and shift
            out = gamma * (x - running_mean) / torch.sqrt(running_var + eps) + beta

            ## ref: https://github.com/AndreiKeino/EECS-498-007-598-005-Deep-Learning-for-Computer-Vision/blob/master/assignments/3/convolutional_networks_completed.ipynb
            ##################################################################
            #                        END OF YOUR CODE                        #
            ##################################################################
        else:
            raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

        # Store the updated running means back into bn_param
        bn_param['running_mean'] = running_mean.detach()
        bn_param['running_var'] = running_var.detach()

        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        Backward pass for batch normalization.

        For this implementation, you should write out a
        computation graph for batch normalization on paper and
        propagate gradients backward through intermediate nodes.

        Inputs:
        - dout: Upstream derivatives, of shape (N, D)
        - cache: Variable of intermediates from batchnorm_forward.

        Returns a tuple of:
        - dx: Gradient with respect to inputs x, of shape (N, D)
        - dgamma: Gradient with respect to scale parameter gamma,
          of shape (D,)
        - dbeta: Gradient with respect to shift parameter beta,
          of shape (D,)
        """
        dx, dgamma, dbeta = None, None, None
        ######################################################################
        # TODO: Implement the backward pass for batch normalization.         #
        # Store the results in the dx, dgamma, and dbeta variables.          #
        # Referencing the original paper (https://arxiv.org/abs/1502.03167)  #
        # might prove to be helpful.                                         #
        # Don't forget to implement train and test mode separately.          #
        ######################################################################
        # Replace "pass" with your code (do not modify this line)
        N, D = dout.shape
        xhat, gamma, subtraction, sdinv, sd, var, eps = cache
        
        # Graphical computation about scale and shift
        dbeta = torch.sum(dout, dim=0)  # upstream gradt: dout // local grad: vector 1
        dgamma = torch.sum(dout * xhat, dim=0)  # upstream grad: dout // local grad: xhat (because of mul gate)
        dxhat = dout * gamma  # upstream grad: dout // local grad: gamma (because of mul gate)
        
        # Graphical computation for the rest
        dsdinv = torch.sum(dxhat * subtraction, dim=0)   # upstream: dxhat // local: subtraction (because of mul gate)
        dsubtraction1 = sdinv * dxhat   # upstream: dxhat // local: sdinv (because of mul gate)
        dsd = -dsdinv / sd ** 2  # upstream: dsdinv // local: -1/sd**2 
        dvar = 0.5 * dsd / torch.sqrt(var + eps)  # upstream: dsd // local: 0.5 * (var + eps)^(-1/2)
        dsq = torch.ones_like(dout) * dvar / (N - 1)  # upstream: dvar // local: 1 / (N-1) * matrix, which filled with and has same size as dout
        dsubtraction2 = 2 * dsq * subtraction   # upstream: dsq // local: 2 * subtraction
        dx1 = dsubtraction1 + dsubtraction2  # Copy gate => add
        dmu = -torch.sum(dx1, dim=0)  # upstream: dx1 // local: - One vector
        dx2 = torch.ones_like(dout) * dmu / N # upstream: dmu // local: matrix filled with one (size: same as dout)
        dx = dx1 + dx2    

        ## ref: https://github.com/AndreiKeino/EECS-498-007-598-005-Deep-Learning-for-Computer-Vision/blob/master/assignments/3/convolutional_networks_completed.ipynb    
        ######################################################################
        #                          END OF YOUR CODE                          #
        ######################################################################

        return dx, dgamma, dbeta

    @staticmethod
    def backward_alt(dout, cache):
        """
        Alternative backward pass for batch normalization.
        For this implementation you should work out the derivatives
        for the batch normalization backward pass on paper and simplify
        as much as possible. You should be able to derive a simple expression
        for the backward pass. See the jupyter notebook for more hints.

        Note: This implementation should expect to receive the same
        cache variable as batchnorm_backward, but might not use all of
        the values in the cache.

        Inputs / outputs: Same as batchnorm_backward
        """
        dx, dgamma, dbeta = None, None, None
        ######################################################################
        # TODO: Implement the backward pass for batch normalization.         #
        # Store the results in the dx, dgamma, and dbeta variables.          #
        #                                                                    #
        # Note: after computing the gradient with respect to the centered    #
        # inputs, gradients with respect to the inputs (dx) can be written   #
        # in a single statement; our implementation fits on a single         #
        # 80-character line. But, it is okay to write it in multiple lines.  #
        ######################################################################
        # Replace "pass" with your code (do not modify this line)
        N, D = dout.shape
        xhat, gamma, subtraction, sdinv, sd, var, eps = cache

        # Graphical computation about scale and shift
        dbeta = torch.sum(dout, dim=0)  # upstream gradt: dout // local grad: vector 1
        dgamma = torch.sum(dout * xhat, dim=0)  # upstream grad: dout // local grad: xhat (because of mul gate)

        # Graphical computation for the rest
        # - first component: gamma * (normalized ) / sd 
        #       -> gamma * ((dout - torch.sum(dout, dim=0)) / N) / sd
        # - second component: subtraction /sd / (N-1) * One matrix / s ** 2 
        dx = gamma * ((dout - torch.sum(dout, dim=0)) / N) / sd - \
             subtraction * torch.sum(dout * subtraction, dim=0) / (sd ** 3 * (N - 1))
        
        ## ref: https://github.com/AndreiKeino/EECS-498-007-598-005-Deep-Learning-for-Computer-Vision/blob/master/assignments/3/convolutional_networks_completed.ipynb
        ######################################################################
        #                          END OF YOUR CODE                          #
        ######################################################################

        return dx, dgamma, dbeta


class SpatialBatchNorm(object):

    @staticmethod
    def forward(x, gamma, beta, bn_param):
        """
        Computes the forward pass for spatial batch normalization.

        Inputs:
        - x: Input data of shape (N, C, H, W)
        - gamma: Scale parameter, of shape (C,)
        - beta: Shift parameter, of shape (C,)
        - bn_param: Dictionary with the following keys:
          - mode: 'train' or 'test'; required
          - eps: Constant for numeric stability
          - momentum: Constant for running mean / variance. momentum=0
            means that old information is discarded completely at every
            time step, while momentum=1 means that new information is never
            incorporated. The default of momentum=0.9 should work well
            in most situations.
          - running_mean: Array of shape (C,) giving running mean of
            features
          - running_var Array of shape (C,) giving running variance
            of features

        Returns a tuple of:
        - out: Output data, of shape (N, C, H, W)
        - cache: Values needed for the backward pass
        """
        out, cache = None, None

        ######################################################################
        # TODO: Implement the forward pass for spatial batch normalization.  #
        # You should implement this by calling the 1D batch normalization    #
        # you implemented above with permuting and/or reshaping input/output #
        # tensors. Your implementation should be very short;                 #
        # less than five lines are expected.                                 #
        ######################################################################
        # Replace "pass" with your code (do not modify this line)
        N, C, H, W = x.shape

        # Since BatchNorm.forward() is normalize across the minibatch dimension N,
        # to do normalization across C channel using BatchNorm.forward()
        # - we should transpose x 
        # - reshape as (C, -1)
        # - transpose C and N again
        x1 = x.contiguous().transpose(0, 1).contiguous().view(C, -1).transpose(0, 1)
        y, cache = BatchNorm.forward(x1, gamma, beta, bn_param)
        # Make out dimension as (N, C, H, W)
        out = y.contiguous().transpose(0, 1).view(C, N, H, W).transpose(0, 1).contiguous()

        ## ref: https://github.com/AndreiKeino/EECS-498-007-598-005-Deep-Learning-for-Computer-Vision/blob/master/assignments/3/convolutional_networks_completed.ipynb
        ######################################################################
        #                          END OF YOUR CODE                          #
        ######################################################################

        return out, cache

    @staticmethod
    def backward(dout, cache):
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

        ######################################################################
        # TODO: Implement the backward pass for spatial batch normalization. #
        # You should implement this by calling the 1D batch normalization    #
        # you implemented above with permuting and/or reshaping input/output #
        # tensors. Your implementation should be very short;                 #
        # less than five lines are expected.                                 #
        ######################################################################
        # Replace "pass" with your code (do not modify this line)
        N, C, H, W = dout.shape
        # Since BatchNorm.backward() is applied across the minibatch dimension N,
        # to do backward about C channel (not N) using BatchNorm.backward()
        # - we should transpose x 
        # - reshape as (C, -1)
        # - transpose C and N again
        dout1 = dout.contiguous().transpose(0, 1).contiguous().view(C, -1).transpose(0, 1)
        dx1, dgamma, dbeta = BatchNorm.backward(dout1, cache)
        # Make out dimension as (N, C, H, W)
        dx = dx1.contiguous().transpose(0, 1).view(C, N, H, W).transpose(0, 1).contiguous()

        ## ref: https://github.com/AndreiKeino/EECS-498-007-598-005-Deep-Learning-for-Computer-Vision/blob/master/assignments/3/convolutional_networks_completed.ipynb
        ######################################################################
        #                          END OF YOUR CODE                          #
        ######################################################################

        return dx, dgamma, dbeta


##################################################################
#            Fast Implementations and Sandwich Layers            #
##################################################################


class Linear(object):

    @staticmethod
    def forward(x, w, b):
        layer = torch.nn.Linear(*w.shape)
        layer.weight = torch.nn.Parameter(w.T)
        layer.bias = torch.nn.Parameter(b)
        tx = x.detach()
        tx.requires_grad = True
        out = layer(tx.flatten(start_dim=1))
        cache = (x, w, b, tx, out, layer)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        try:
            x, w, b, tx, out, layer = cache
            out.backward(dout)
            dx = tx.grad.detach()
            dw = layer.weight.grad.detach().T
            db = layer.bias.grad.detach()
            layer.weight.grad = layer.bias.grad = None
        except RuntimeError:
            dx = torch.zeros_like(tx)
            dw = torch.zeros_like(layer.weight).T
            db = torch.zeros_like(layer.bias)
        return dx, dw, db


class ReLU(object):

    @staticmethod
    def forward(x):
        layer = torch.nn.ReLU()
        tx = x.detach()
        tx.requires_grad = True
        out = layer(tx)
        cache = (x, tx, out, layer)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        try:
            x, tx, out, layer = cache
            out.backward(dout)
            dx = tx.grad.detach()
        except RuntimeError:
            dx = torch.zeros_like(tx)
        return dx


class Linear_ReLU(object):

    @staticmethod
    def forward(x, w, b):
        """
        Convenience layer that performs a linear transform followed by a ReLU.

        Inputs:
        - x: Input to the linear layer
        - w, b: Weights for the linear layer
        Returns a tuple of:
        - out: Output of the ReLU
        - cache: Object to give to the backward pass
        """
        a, fc_cache = Linear.forward(x, w, b)
        out, relu_cache = ReLU.forward(a)
        cache = (fc_cache, relu_cache)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        Backward pass for the linear-relu convenience layer
        """
        fc_cache, relu_cache = cache
        da = ReLU.backward(dout, relu_cache)
        dx, dw, db = Linear.backward(da, fc_cache)
        return dx, dw, db


class FastConv(object):

    @staticmethod
    def forward(x, w, b, conv_param):
        N, C, H, W = x.shape
        F, _, HH, WW = w.shape
        stride, pad = conv_param['stride'], conv_param['pad']
        layer = torch.nn.Conv2d(C, F, (HH, WW), stride=stride, padding=pad)
        layer.weight = torch.nn.Parameter(w)
        layer.bias = torch.nn.Parameter(b)
        tx = x.detach()
        tx.requires_grad = True
        out = layer(tx)
        cache = (x, w, b, conv_param, tx, out, layer)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        try:
            x, _, _, _, tx, out, layer = cache
            out.backward(dout)
            dx = tx.grad.detach()
            dw = layer.weight.grad.detach()
            db = layer.bias.grad.detach()
            layer.weight.grad = layer.bias.grad = None
        except RuntimeError:
            dx = torch.zeros_like(tx)
            dw = torch.zeros_like(layer.weight)
            db = torch.zeros_like(layer.bias)
        return dx, dw, db


class FastMaxPool(object):

    @staticmethod
    def forward(x, pool_param):
        N, C, H, W = x.shape
        pool_height = pool_param['pool_height']
        pool_width = pool_param['pool_width']
        stride = pool_param['stride']
        layer = torch.nn.MaxPool2d(kernel_size=(pool_height, pool_width),
                                   stride=stride)
        tx = x.detach()
        tx.requires_grad = True
        out = layer(tx)
        cache = (x, pool_param, tx, out, layer)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        try:
            x, _, tx, out, layer = cache
            out.backward(dout)
            dx = tx.grad.detach()
        except RuntimeError:
            dx = torch.zeros_like(tx)
        return dx


class Conv_ReLU(object):

    @staticmethod
    def forward(x, w, b, conv_param):
        """
        A convenience layer that performs a convolution
        followed by a ReLU.
        Inputs:
        - x: Input to the convolutional layer
        - w, b, conv_param: Weights and parameters for the
          convolutional layer
        Returns a tuple of:
        - out: Output from the ReLU
        - cache: Object to give to the backward pass
        """
        a, conv_cache = FastConv.forward(x, w, b, conv_param)
        out, relu_cache = ReLU.forward(a)
        cache = (conv_cache, relu_cache)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        Backward pass for the conv-relu convenience layer.
        """
        conv_cache, relu_cache = cache
        da = ReLU.backward(dout, relu_cache)
        dx, dw, db = FastConv.backward(da, conv_cache)
        return dx, dw, db


class Conv_ReLU_Pool(object):

    @staticmethod
    def forward(x, w, b, conv_param, pool_param):
        """
        A convenience layer that performs a convolution,
        a ReLU, and a pool.
        Inputs:
        - x: Input to the convolutional layer
        - w, b, conv_param: Weights and parameters for
          the convolutional layer
        - pool_param: Parameters for the pooling layer
        Returns a tuple of:
        - out: Output from the pooling layer
        - cache: Object to give to the backward pass
        """
        a, conv_cache = FastConv.forward(x, w, b, conv_param)
        s, relu_cache = ReLU.forward(a)
        out, pool_cache = FastMaxPool.forward(s, pool_param)
        cache = (conv_cache, relu_cache, pool_cache)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        Backward pass for the conv-relu-pool convenience layer.
        """
        conv_cache, relu_cache, pool_cache = cache
        ds = FastMaxPool.backward(dout, pool_cache)
        da = ReLU.backward(ds, relu_cache)
        dx, dw, db = FastConv.backward(da, conv_cache)
        return dx, dw, db


class FastBatchNorm(object):
    func = torch.nn.BatchNorm1d

    @classmethod
    def forward(cls, x, gamma, beta, bn_param):
        mode = bn_param['mode']
        eps = bn_param.get('eps', 1e-5)
        momentum = bn_param.get('momentum', 0.9)
        D = x.shape[1]
        running_mean = \
            bn_param.get('running_mean',
                         torch.zeros(D, dtype=x.dtype, device=x.device))
        running_var = \
            bn_param.get('running_var',
                         torch.ones(D, dtype=x.dtype, device=x.device))

        layer = cls.func(D, eps=eps, momentum=momentum,
                         device=x.device, dtype=x.dtype)
        layer.weight = torch.nn.Parameter(gamma)
        layer.bias = torch.nn.Parameter(beta)
        layer.running_mean = running_mean
        layer.running_var = running_var
        if mode == 'train':
            layer.train()
        elif mode == 'test':
            layer.eval()
        else:
            raise ValueError('Invalid forward batchnorm mode "%s"' % mode)
        tx = x.detach()
        tx.requires_grad = True
        out = layer(tx)
        cache = (mode, x, tx, out, layer)
        # Store the updated running means back into bn_param
        bn_param['running_mean'] = layer.running_mean.detach()
        bn_param['running_var'] = layer.running_var.detach()
        return out, cache

    @classmethod
    def backward(cls, dout, cache):
        mode, x, tx, out, layer = cache
        try:
            if mode == 'train':
                layer.train()
            elif mode == 'test':
                layer.eval()
            else:
                raise ValueError('Invalid forward batchnorm mode "%s"' % mode)
            out.backward(dout)
            dx = tx.grad.detach()
            dgamma = layer.weight.grad.detach()
            dbeta = layer.bias.grad.detach()
            layer.weight.grad = layer.bias.grad = None
        except RuntimeError:
            dx = torch.zeros_like(tx)
            dgamma = torch.zeros_like(layer.weight)
            dbeta = torch.zeros_like(layer.bias)
        return dx, dgamma, dbeta


class FastSpatialBatchNorm(FastBatchNorm):
    func = torch.nn.BatchNorm2d


class Linear_BatchNorm_ReLU(object):

    @staticmethod
    def forward(x, w, b, gamma, beta, bn_param):
        """
        Convenience layer that performs an linear transform,
        batch normalization, and ReLU.
        Inputs:
        - x: Array of shape (N, D1); input to the linear layer
        - w, b: Arrays of shape (D1, D2) and (D2,) giving the
          weight and bias for the linear transform.
        - gamma, beta: Arrays of shape (D2,) and (D2,) giving
          scale and shift parameters for batch normalization.
        - bn_param: Dictionary of parameters for batch
          normalization.
        Returns:
        - out: Output from ReLU, of shape (N, D2)
        - cache: Object to give to the backward pass.
        """
        a, fc_cache = Linear.forward(x, w, b)
        a_bn, bn_cache = FastBatchNorm.forward(a, gamma, beta, bn_param)
        out, relu_cache = ReLU.forward(a_bn)
        cache = (fc_cache, bn_cache, relu_cache)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        Backward pass for the linear-batchnorm-relu convenience layer.
        """
        fc_cache, bn_cache, relu_cache = cache
        da_bn = ReLU.backward(dout, relu_cache)
        da, dgamma, dbeta = FastBatchNorm.backward(da_bn, bn_cache)
        dx, dw, db = Linear.backward(da, fc_cache)
        return dx, dw, db, dgamma, dbeta


class Conv_BatchNorm_ReLU(object):

    @staticmethod
    def forward(x, w, b, gamma, beta, conv_param, bn_param):
        a, conv_cache = FastConv.forward(x, w, b, conv_param)
        an, bn_cache = FastSpatialBatchNorm.forward(a, gamma,
                                                    beta, bn_param)
        out, relu_cache = ReLU.forward(an)
        cache = (conv_cache, bn_cache, relu_cache)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        conv_cache, bn_cache, relu_cache = cache
        dan = ReLU.backward(dout, relu_cache)
        da, dgamma, dbeta = FastSpatialBatchNorm.backward(dan, bn_cache)
        dx, dw, db = FastConv.backward(da, conv_cache)
        return dx, dw, db, dgamma, dbeta


class Conv_BatchNorm_ReLU_Pool(object):

    @staticmethod
    def forward(x, w, b, gamma, beta, conv_param, bn_param, pool_param):
        a, conv_cache = FastConv.forward(x, w, b, conv_param)
        an, bn_cache = FastSpatialBatchNorm.forward(a, gamma, beta, bn_param)
        s, relu_cache = ReLU.forward(an)
        out, pool_cache = FastMaxPool.forward(s, pool_param)
        cache = (conv_cache, bn_cache, relu_cache, pool_cache)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        conv_cache, bn_cache, relu_cache, pool_cache = cache
        ds = FastMaxPool.backward(dout, pool_cache)
        dan = ReLU.backward(ds, relu_cache)
        da, dgamma, dbeta = FastSpatialBatchNorm.backward(dan, bn_cache)
        dx, dw, db = FastConv.backward(da, conv_cache)
        return dx, dw, db, dgamma, dbeta
