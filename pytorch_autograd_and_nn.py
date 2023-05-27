"""
Implements pytorch autograd and nn in PyTorch.
WARNING: you SHOULD NOT use ".to()" or ".cuda()" in each implementation block.
"""
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def hello():
    """
    This is a sample function that we will try to import and run to ensure
    that our environment is correctly set up on Google Colab.
    """
    print('Hello from pytorch_autograd_and_nn.py!')


##############################################################################
# Part 1. Barebones PyTorch                                                  #
##############################################################################


def three_layer_convnet(x, params, hyperparams):
    """
    Performs the forward pass of a three-layer convolutional network with the
    architecture defined above.

    Inputs:
    - x: Tensor of shape (N, C, H, W) giving a minibatch of images
    - params: List of Tensors giving the weights and biases for the network;
      should contain the following:
        - conv_w1: Tensor of shape (channel_1, C, KH1, KW1) giving weights for
          the first convolutional layer.
        - conv_b1: Tensor of shape (channel_1,) giving biases for the first
          convolutional layer.
        - conv_w2: Tensor of shape (channel_2, channel_1, KH2, KW2) giving
          weights for the second convolutional layer.
        - conv_b2: Tensor of shape (channel_2,) giving biases for the second
          convolutional layer.
        - fc_w: Tensor giving weights for the fully-connected layer.
        - fc_b: Tensor giving biases for the fully-connected layer.
    - hyperparams: Dictionary of hyperparameters for the network;
      should contain the following:
        - padding: List of the padding size applied for each layer.
    Returns:
    - scores: Tensor of shape (N, C) giving classification scores for x
    """
    conv_w1, conv_b1, conv_w2, conv_b2, fc_w, fc_b = params
    pad_size_1, pad_size_2 = hyperparams['padding']
    scores = None
    ##########################################################################
    # TODO: Implement the forward pass for the three-layer ConvNet.          #
    # The network has the following architecture:                            #
    #                                                                        #
    # 1. Conv layer (with bias) with given filters and zero-padding          #
    # 2. ReLU                                                                #
    # 3. Conv layer (with bias) with given filters and zero-padding          #
    # 4. ReLU                                                                #
    # 5. Fully-connected layer (with bias) to compute scores                 #
    #                                                                        #
    # For simplicity, yon can assume same padding for all conv layers.       #
    # Hint: F.linear, F.conv2d, F.relu, torch.flatten                        #
    ##########################################################################
    # Replace "pass" with your code (do not modify this line)
    h1 = F.relu(F.conv2d(x, weight=conv_w1, bias=conv_b1, padding=pad_size_1))
    h2 = F.relu(F.conv2d(h1, weight=conv_w2, bias=conv_b2, padding=pad_size_2))
    scores = F.linear(torch.flatten(h2, start_dim=1), fc_w, fc_b)
    ##########################################################################
    #                            END OF YOUR CODE                            #
    ##########################################################################
    return scores


def initialize_three_layer_conv_part1(dtype=torch.float, device='cpu'):
    '''
    Initializes weights for the three_layer_convnet for part 1
    Inputs:
    - dtype: Torch data type object; all computations will be performed.
      using this dtype. float is faster but less accurate, so you should use
      double when numeric gradient checking.
    - device: Device to use for computation. 'cpu' or 'cuda'
    '''
    # Input/Output dimensions
    C, H, W = 3, 32, 32
    num_classes = 10

    # Hidden layer channel and kernel sizes
    channel_1 = 12
    channel_2 = 8
    kernel_size_1 = 5
    kernel_size_2 = 3
    pad_size_1 = 2
    pad_size_2 = 1

    # Initialize the weights
    conv_w1 = None
    conv_b1 = None
    conv_w2 = None
    conv_b2 = None
    fc_w = None
    fc_b = None

    ##########################################################################
    # TODO: Define and initialize the parameters of a three-layer ConvNet    #
    # using nn.init.kaiming_normal_. You should initialize your bias vectors #
    # using the zero_weight function. You are given all the necessary        #
    # variables above for initializing weights.                              #
    ##########################################################################
    # Replace "pass" with your code (do not modify this line)
    conv_w1 = nn.init.kaiming_normal_(torch.empty((channel_1, C, kernel_size_1, kernel_size_1), dtype=dtype, device=device, requires_grad=True))
    conv_b1 = nn.init.zeros_(torch.empty(channel_1, dtype=dtype, device=device, requires_grad=True))
    conv_w2 = nn.init.kaiming_normal_(torch.empty((channel_2, channel_1, kernel_size_2, kernel_size_2), dtype=dtype, device=device, requires_grad=True))
    conv_b2 = nn.init.zeros_(torch.empty(channel_2, dtype=dtype, device=device, requires_grad=True))
    fc_w = nn.init.kaiming_normal_(torch.empty((num_classes, channel_2 * 32 * 32), dtype=dtype, device=device, requires_grad=True))
    fc_b = nn.init.zeros_(torch.empty(num_classes, dtype=dtype, device=device, requires_grad=True))
    ##########################################################################
    #                            END OF YOUR CODE                            #
    ##########################################################################
    params = [conv_w1, conv_b1, conv_w2, conv_b2, fc_w, fc_b]
    hyperparams = {'padding': [pad_size_1, pad_size_2]}
    return params, hyperparams


##############################################################################
# Part 2. PyTorch Module API                                                 #
##############################################################################

class ThreeLayerConvNet(nn.Module):
    """
    An implementation of three-layer convolutional network.
    """
    def __init__(self, in_shape, conv_params, num_classes):
        """
        Initialize a new network.
        Inputs:
        - in_shape: A tuple of (C, H, W), where
            - C: Number of input channels
            - H: Height of input
            - W: Width of input
        - conv_params: List of convolutional layers, where each item is
          a tuple of (num_filters, kernel_size, pad_size):
            - num_filters: Number of filters to use in the convolutional layer
            - kernel_size: Width/Height of filters
            - pad_size: Size of padding applied to the input
        - num_classes: Number of scores to produce from the final linear layer
        """
        super().__init__()

        C, H, W = in_shape
        channel_1, kernel_size_1, pad_size_1 = conv_params[0]
        channel_2, kernel_size_2, pad_size_2 = conv_params[1]
        ######################################################################
        # TODO: Set up layers with learnable parameters for a three-layer    #
        # ConvNet with the architecture defined below. You should initialize #
        # the weights of the model using the Kaiming normal initialization,  #
        # and zero out the bias vectors.                                     #
        # The architecture should be similar to Part 1:                      #
        #                                                                    #
        # 1. self.conv1: Conv layer with channel_1 filters, each with shape  #
        #    [kernel_size_1, kernel_size_1], and zero-padding of pad_size_1  #
        # 2. ReLU                                                            #
        # 3. self.conv2: Conv layer with channel_2 filters, each with shape  #
        #    [kernel_size_2, kernel_size_2], and zero-padding of pad_size_2  #
        # 4. ReLU                                                            #
        # 5. self.fc: Fully-connected layer to num_classes classes           #
        #                                                                    #
        # For simplicity, yon can assume same padding for all conv layers.   #
        # Hint: nn.Linear, nn.Conv2d, nn.init.kaiming_normal_,               #
        #       nn.init.zeros_                                               #
        ######################################################################
        # Replace "pass" with your code (do not modify this line)
        # Define layers with learnable parameters
        self.conv1 = nn.Conv2d(C, channel_1, kernel_size_1, padding=pad_size_1)
        self.conv2 = nn.Conv2d(channel_1, channel_2, kernel_size_2, padding=pad_size_2)
        self.fc = nn.Linear(channel_2 * H * W, num_classes)

        # Initialize
        conv_w1 = nn.init.kaiming_normal_(self.conv1.weight)
        conv_b1 = nn.init.zeros_(self.conv1.bias)
        conv_w2 = nn.init.kaiming_normal_(self.conv2.weight)
        conv_b2 = nn.init.zeros_(self.conv2.bias)
        fc_w = nn.init.kaiming_normal_(self.fc.weight)
        fc_b = nn.init.zeros_(self.fc.bias)
        ######################################################################
        #                          END OF YOUR CODE                          #
        ######################################################################

    def forward(self, x):
        scores = None
        ######################################################################
        # TODO: Implement the forward function for a three-layer ConvNet.    #
        # you should use the layers you defined in __init__ and specify the  #
        # connectivity of those layers in forward().                         #
        # Hint: F.relu, torch.flatten                                        #
        ######################################################################
        # Replace "pass" with your code (do not modify this line)
        h1 = F.relu(self.conv1(x))
        h2 = F.relu(self.conv2(h1))
        scores = self.fc(torch.flatten(h2, start_dim=1))
        ######################################################################
        #                          END OF YOUR CODE                          #
        ######################################################################
        return scores


def initialize_three_layer_conv_part2():
    '''
    Instantiates a ThreeLayerConvNet model and a corresponding optimizer
    for part 2
    '''

    # Input/Output dimensions
    C, H, W = 3, 32, 32
    num_classes = 10

    # Hidden layer channel and kernel sizes
    channel_1 = 12
    channel_2 = 8
    kernel_size_1 = 5
    kernel_size_2 = 3
    pad_size_1 = 2
    pad_size_2 = 1

    # Parameters for optimizer
    lr = 1e-2
    momentum = 0.5
    weight_decay = 1e-4

    model = None
    optimizer = None
    ###########################################################################
    # TODO: Instantiate ThreeLayerConvNet model and a corresponding optimizer.#
    # Use the above mentioned variables for setting the parameters.           #
    # You should train the model using stochastic gradient descent with       #
    # momentum and L2 weight decay.                                           #
    # Hint: optim.SGD                                                         #
    ###########################################################################
    # Replace "pass" with your code (do not modify this line)
    in_shape = (C, H, W)
    conv_params = [(channel_1, kernel_size_1, pad_size_1), (channel_2, kernel_size_2, pad_size_2)]
    model = ThreeLayerConvNet(in_shape, conv_params, num_classes)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    ##########################################################################
    #                            END OF YOUR CODE                            #
    ##########################################################################
    return model, optimizer


##############################################################################
# Part 3. PyTorch Sequential API                                             #
##############################################################################

def initialize_three_layer_conv_part3():
    '''
    Instantiates a ThreeLayerConvNet model and a corresponding optimizer
    for part 3
    '''
    # Input/Output dimensions
    C, H, W = 3, 32, 32
    num_classes = 10

    # Hidden layer channel and kernel sizes
    channel_1 = 12
    channel_2 = 8
    kernel_size_1 = 5
    kernel_size_2 = 3
    pad_size_1 = 2
    pad_size_2 = 1

    # Parameters for optimizer
    lr = 1e-3
    weight_decay = 1e-4

    model = None
    optimizer = None
    ##########################################################################
    # TODO: Rewrite the three-layer ConvNet (with bias) from Part 2 with     #
    # Sequential API and a corresponding optimizer.                          #
    # You should use `nn.Sequential` to define a three-layer ConvNet with    #
    # named modules as follows:                                              #
    #                                                                        #
    # 1. conv1: Conv layer with channel_1 filters, each with shape           #
    #    [kernel_size_1, kernel_size_1], and zero-padding of pad_size_1      #
    # 2. relu1: ReLU                                                         #
    # 3. conv2: Conv layer with channel_2 filters, each with shape           #
    #    [kernel_size_2, kernel_size_2], and zero-padding of pad_size_2      #
    # 4. relu2: ReLU                                                         #
    # 5. flatten: Flatten                                                    #
    # 6. fc: Fully-connected layer to num_classes classes                    #
    #                                                                        #
    # For simplicity, yon can assume same padding for all conv layers.       #
    # To see how to give a name to each module, you may want to take a look  #
    # at the example of Two-Layer Network in ipynb                           #
    # You should initialize the weights of the model using the Kaiming       #
    # normal initialization and zero out the bias vectors. PyTorch Module    #
    # class provides an API .modules() to iterate over all modules in there. #
    # To see how to initialize each module, you may want to take a look at   #
    # the example of Two-Layer Network in ipynb                              #
    # You should optimize your model using adam optimizer with L2 weight     #
    # decay with variables given above.                                      #
    # Hint: OrderedDict, nn.Sequential, nn.ReLU, nn.Flatten                  #
    ##########################################################################
    # Replace "pass" with your code (do not modify this line)
    # Define model with nn.Sequential
    model = nn.Sequential(OrderedDict([
        ('conv1', nn.Conv2d(C, channel_1, kernel_size_1, padding=pad_size_1)),
        ('relu1', nn.ReLU()),
        ('conv2', nn.Conv2d(channel_1, channel_2, kernel_size_2, padding=pad_size_2)),
        ('relu2', nn.ReLU()),
        ('flatten', nn.Flatten()),
        ('fc', nn.Linear(channel_2 * H * W, num_classes))
    ]))

    # Initialize
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            nn.init.zeros_(m.bias)

    # Optimize
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    ##########################################################################
    #                            END OF YOUR CODE                            #
    ##########################################################################
    return model, optimizer


##############################################################################
# Part 4. ResNet for CIFAR-10                                                #
##############################################################################

class PlainBlock(nn.Module):

    expansion: int = 1

    def __init__(self, Cin, Cout, downsample=False):
        """
        Initialize a plain block.
        Inputs:
        - Cin: Number of input channels
        - Cout: Number of output channels
        - downsample: Add downsampling (a conv with stride=2) if True
        """
        super().__init__()

        self.net = None
        ######################################################################
        # TODO: Implement plain block with the following architecture:       #
        #                                                                    #
        # 1. Spatial Batch normalization with Cin channels                   #
        # 2. ReLU                                                            #
        # 3. Conv layer (without bias) with Cout 3x3 filters, zero-padding   #
        #    of 1, and stride 2 if downsampling; otherwise stride 1          #
        # 4. Spatial Batch normalization with Cout channels                  #
        # 5. ReLU                                                            #
        # 6. Conv layer (without bias) with Cout 3x3 filters and             #
        #    zero-padding of 1                                               #
        #                                                                    #
        # Store the result in self.net                                       #
        # Hint: Wrap your layers by nn.Sequential to output a single module. #
        ######################################################################
        # Replace "pass" with your code (do not modify this line)
        kernel_size = 3
        pad_size = 1
        if downsample:
            stride = 2
        else:
            stride = 1
        
        self.net = nn.Sequential(
            nn.BatchNorm2d(Cin),
            nn.ReLU(),
            nn.Conv2d(Cin, Cout, kernel_size, padding=pad_size, stride=stride, bias=False),
            nn.BatchNorm2d(Cout),
            nn.ReLU(),
            nn.Conv2d(Cout, Cout, kernel_size, padding=pad_size, bias=False)
        )
        ######################################################################
        #                          END OF YOUR CODE                          #
        ######################################################################

    def forward(self, x):
        return self.net(x)


class ResidualBlock(nn.Module):

    expansion: int = 1

    def __init__(self, Cin, Cout, downsample=False):
        """
        Initialize a residual block.

        Inputs:
        - Cin: Number of input channels
        - Cout: Number of output channels
        - downsample: Add downsampling (a conv with stride=2) if True
        """
        super().__init__()

        self.block = None  # F
        self.shortcut = None  # G
        ######################################################################
        # TODO: Implement residual block. Store the main block in self.block #
        # and the shortcut in self.shortcut. Use PlainBlock for the main     #
        # block for simplicity.                                              #
        # Hint: nn.Identity() for identity mapping                           #
        ######################################################################
        # Replace "pass" with your code (do not modify this line)
        self.block = nn.Sequential(PlainBlock(Cin, Cout, downsample))
        # if downsample == True:
        #     # downsampling
        #     self.shortcut = nn.Sequential(nn.Conv2d(Cin, Cout, 1, stride=2, bias=False))
        # elif Cin == Cout:
        #     # no downsampling and Cin==Cout
        #     self.shortcut = nn.Sequential(nn.Identity())
        # else:
        #     # no downsampling and Cin != Cout
        #     self.shortcut == nn.Sequential(nn.Conv2d(Cin, Cout, 1, stride=1, bias=False))
        if (Cin == Cout) and (downsample == False):
            self.shortcut = nn.Sequential(nn.Identity())
        elif (Cin != Cout) and (downsample == False):
            self.shortcut = nn.Sequential(nn.Conv2d(Cin, Cout, kernel_size=1, stride=1, bias=False))
        else:
            self.shortcut = nn.Sequential(nn.Conv2d(Cin, Cout, kernel_size=1, stride=2, bias=False))
        ######################################################################
        #                          END OF YOUR CODE                          #
        ######################################################################

    def forward(self, x):
        return self.block(x) + self.shortcut(x)


class ResNetStage(nn.Module):
    def __init__(self, Cin, Cout, num_blocks, downsample=True,
                 block=ResidualBlock):
        super().__init__()
        blocks = [block(Cin, Cout, downsample)]
        for _ in range(num_blocks - 1):
            blocks.append(block(Cout * block.expansion, Cout))
        self.net = nn.Sequential(*blocks)

    def forward(self, x):
        return self.net(x)


class ResNetStem(nn.Module):
    def __init__(self, Cin=3, Cout=16):
        super().__init__()
        self.net = nn.Conv2d(Cin, Cout, kernel_size=3, stride=1, padding=1,
                             bias=False)

    def forward(self, x):
        return self.net(x)


class ResNet(nn.Module):
    def __init__(self, stage_args, in_channels=3, block=ResidualBlock,
                 num_classes=10):
        """
        Initialize a residual network. The architecture should look like:

        stem - [block] * N - batchnorm - relu - avgpool - fc

        Inputs:
        - stage_args: A tuple of (C, num_blocks, downsample), where
            - C: Number of channels
            - num_blocks: Number of blocks
            - downsample: Add downsampling (a conv with stride=2) if True
        - in_channels: Number of input channels for stem
        - block: Class of the building block
        - num_classes: Number of scores to produce from the final linear layer.
        """
        super().__init__()

        self.cnn = None
        ######################################################################
        # TODO: Implement the convolutional part of ResNet using ResNetStem  #
        # and ResNetStage provided above, and wrap the modules by            #
        # nn.Sequential. Store the model in self.cnn.                        #
        # You should multiply block.expansion to the dimension of each stage #
        # input except for the first stage and the last batchnorm to make    #
        # this compatible with ResidualBottleneck.                           #
        # Hint: nn.AdaptiveAvgPool2d((1, 1)) for global average pooling      #
        ######################################################################
        # Replace "pass" with your code (do not modify this line)
        # stem
        blocks = [ResNetStem(in_channels, stage_args[0][0])]

        # [block] * N
        for i, (C, num_blocks, downsample) in enumerate(stage_args):
            if i == 0:
                blocks.append(ResNetStage(C, C, num_blocks, downsample, block)) 
            else:
                C_in = C * block.expansion
                blocks.append(ResNetStage(C_in, C, num_blocks, downsample, block))

        # batchnorm - relu - avgpool
        blocks.append(nn.BatchNorm2d(stage_args[-1][0])) 
        blocks.append(nn.ReLU())
        blocks.append(nn.AdaptiveAvgPool2d((1,1)))

        self.cnn = nn.Sequential(*blocks)
        ######################################################################
        #                          END OF YOUR CODE                          #
        ######################################################################
        self.fc = nn.Linear(stage_args[-1][0] * block.expansion, num_classes)

    def forward(self, x):
        scores = None
        ######################################################################
        # TODO: Implement the forward function of ResNet.                    #
        # Store the output in `scores`.                                      #
        # Hint: this class separates the convolutional part and              #
        # fully-connected part, so you need to add flatten here.             #
        ######################################################################
        # Replace "pass" with your code (do not modify this line)
        scores = self.cnn(x)
        scores = self.fc(torch.flatten(scores, start_dim=1))

        ######################################################################
        #                          END OF YOUR CODE                          #
        ######################################################################
        return scores


class ResidualBottleneck(nn.Module):

    expansion: int = 4

    def __init__(self, Cin, Cout, downsample=False):
        """
        Initialize a residual bottleneck block.
        Inputs:
        - Cin: Number of input channels
        - Cout: Number of output channels
        - downsample: Add downsampling (a conv with stride=2) if True
        """
        super().__init__()

        self.block = None
        self.shortcut = None
        ######################################################################
        # TODO: Implement residual bottleneck block as follows:              #
        #                                                                    #
        # 1. Spatial Batch normalization with Cin channels                   #
        # 2. ReLU                                                            #
        # 3. Conv layer (without bias) with Cout 1x1 filters, stride 2       #
        #    if downsampling; otherwise stride 1                             #
        # 4. Spatial Batch normalization with Cout channels                  #
        # 5. ReLU                                                            #
        # 6. Conv layer (without bias) with Cout 3x3 filters and             #
        #    zero-padding of 1                                               #
        # 7. Spatial Batch normalization with Cout channels                  #
        # 8. ReLU                                                            #
        # 9. Conv layer (without bias) with Cout*4 1x1 filters               #
        #                                                                    #
        # Store the main block in self.block and shortcut in self.shortcut   #
        ######################################################################
        # Replace "pass" with your code (do not modify this line)
        # block part
        if downsample :
            stride = 2
        else:
            stride = 1
        self.block = nn.Sequential(
            nn.BatchNorm2d(Cin),
            nn.ReLU(),
            nn.Conv2d(Cin, Cout, 1, stride=stride, bias=False),
            nn.BatchNorm2d(Cout),
            nn.ReLU(),
            nn.Conv2d(Cout, Cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(Cout),
            nn.ReLU(),
            nn.Conv2d(Cout, Cout * self.expansion, 1, bias=False)
        )
        # shortcut part
        if (Cin == Cout) and (downsample == False):
            self.shortcut = nn.Sequential(nn.Identity())
        elif (Cin != Cout) and (downsample == False):
            self.shortcut = nn.Sequential(nn.Conv2d(Cin, Cout * self.expansion, kernel_size=1, stride=1, bias=False))
        else:
            self.shortcut = nn.Sequential(nn.Conv2d(Cin, Cout * self.expansion, kernel_size=1, stride=2, bias=False))
        ######################################################################
        #                          END OF YOUR CODE                          #
        ######################################################################

    def forward(self, x):
        return self.block(x) + self.shortcut(x)
