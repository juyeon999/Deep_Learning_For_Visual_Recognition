"""
Implements rnn and lstm for image captioning in PyTorch.
WARNING: you SHOULD NOT use ".to()" or ".cuda()" in each implementation block.
"""
import math
from typing import Optional, Tuple

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import feature_extraction


def hello():
    """
    This is a sample function that we will try to import and run to ensure
    that our environment is correctly set up on Google Colab.
    """
    print('Hello from rnn_lstm_captioning.py!')


def rnn_step_forward(x, prev_h, Wx, Wh, b):
    """
    Run the forward pass for a single timestep of a vanilla RNN that uses
    a tanh activation function.

    The input data has dimension D, the hidden state has dimension H, and we
    use a minibatch size of N.

    Args:
    - x: Input data for this timestep, of shape (N, D).
    - prev_h: Hidden state from previous timestep, of shape (N, H)
    - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
    - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
    - b: Biases, of shape (H,)

    Returns a tuple of:
    - next_h: Next hidden state, of shape (N, H)
    - cache: Tuple of values needed for the backward pass.
    """
    next_h, cache = None, None
    ##########################################################################
    # TODO: Implement a single forward step for the vanilla RNN. Store next  #
    # hidden state and any values you need for the backward pass in next_h   #
    # and cache variables respectively.                                      #
    # Hint: torch.tanh                                                       #
    ##########################################################################
    # Replace "pass" with your code (do not modify this line)
    next_h = torch.tanh(torch.mm(prev_h, Wh) + torch.mm(x, Wx) + b)
    cache = (x, prev_h, Wx, Wh, next_h)

    ## ref: No reference(My own code)
    ##########################################################################
    #                            END OF YOUR CODE                            #
    ##########################################################################
    return next_h, cache


def rnn_step_backward(dnext_h, cache):
    """
    Backward pass for a single timestep of a vanilla RNN.

    Args:
    - dnext_h: Gradient of loss with respect to next hidden state,
      of shape (N, H)
    - cache: Cache object from the forward pass

    Returns a tuple of:
    - dx: Gradients of input data, of shape (N, D)
    - dprev_h: Gradients of previous hidden state, of shape (N, H)
    - dWx: Gradients of input-to-hidden weights, of shape (D, H)
    - dWh: Gradients of hidden-to-hidden weights, of shape (H, H)
    - db: Gradients of bias vector, of shape (H,)
    """
    dx, dprev_h, dWx, dWh, db = None, None, None, None, None
    ##########################################################################
    # TODO: Implement a single backward step of a vanilla RNN.               #
    # Hint: d tanh(x) / dx = 1 - [tanh(x)]^2                                 #
    ##########################################################################
    # Replace "pass" with your code (do not modify this line)
    x, prev_h, Wx, Wh, next_h = cache

    dL = dnext_h * (1 - next_h ** 2)      # (N, H)
    dx = torch.mm(dL, Wx.T)  # upstream: dnext_h (N, D)
                             # downstream: torch.mm(dtanh, Wx.T) where dtanh (N, H), Wx (D, H)
                             # * is elementwise multiplication
                             # inner differential: Wx
    dprev_h = torch.mm(dL, Wh.T)  # inner differential: Wh
    dWx = torch.mm(x.T, dL)       # inner differential: x
    dWh = torch.mm(prev_h.T, dL)  # inner differential: prev_h
    db = torch.sum(dL, dim=0)     # b is constant so that summation of all elements of dL
    
    # ref: https://github.com/AndreiKeino/EECS-498-007-598-005-Deep-Learning-for-Computer-Vision/blob/master/assignments/4/rnn_lstm_attention_captioning_completed.ipynb
    ##########################################################################
    #                            END OF YOUR CODE                            #
    ##########################################################################
    return dx, dprev_h, dWx, dWh, db


def rnn_forward(x, h0, Wx, Wh, b):
    """
    Run a vanilla RNN forward on an entire sequence of data. We assume an input
    sequence composed of T vectors, each of dimension D. The RNN uses a hidden
    size of H, and we work over a minibatch containing N sequences. After
    running the RNN forward, we return the hidden states for all timesteps.

    Args:
    - x: Input data for the entire timeseries, of shape (N, T, D).
    - h0: Initial hidden state, of shape (N, H)
    - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
    - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
    - b: Biases, of shape (H,)

    Returns a tuple of:
    - h: Hidden states for the entire timeseries, of shape (N, T, H).
    - cache: Values needed in the backward pass
    """
    h, cache = None, None
    ##########################################################################
    # TODO: Implement the forward pass for a vanilla RNN running on a        #
    # sequence of input data. You should use the rnn_step_forward function   #
    # that you defined above.                                                #
    # Hint: You may want to use a for-loop.                                  #
    ##########################################################################
    # Replace "pass" with your code (do not modify this line)
    N, T, D, = x.shape
    H = Wx.shape[1]

    h = torch.zeros((N, T, H)).to(x)
    cache = []
    prev_h = h0.clone()  # for iteration

    for t in range(T):
        xt = x[:, t, :]
        prev_h, cache_t = rnn_step_forward(xt, prev_h, Wx, Wh, b)

        h[:, t, :] = prev_h
        cache.append(cache_t)

    ## ref: My own code (No reference)
    ##########################################################################
    #                            END OF YOUR CODE                            #
    ##########################################################################
    return h, cache


def rnn_backward(dh, cache):
    """
    Backward pass for a vanilla RNN over an entire sequence of data.

    Args:
    - dh: Upstream gradients of all hidden states, of shape (N, T, H).

    NOTE: 'dh' contains the upstream gradients produced by the
    individual loss functions at each timestep, *not* the gradients
    being passed between timesteps (which you'll have to compute yourself
    by calling rnn_step_backward in a loop).

    Returns a tuple of:
    - dx: Gradient of inputs, of shape (N, T, D)
    - dh0: Gradient of initial hidden state, of shape (N, H)
    - dWx: Gradient of input-to-hidden weights, of shape (D, H)
    - dWh: Gradient of hidden-to-hidden weights, of shape (H, H)
    - db: Gradient of biases, of shape (H,)
    """
    dx, dh0, dWx, dWh, db = None, None, None, None, None
    ##########################################################################
    # TODO: Implement the backward pass for a vanilla RNN running on a       #
    # sequence of input data. You should use the rnn_step_backward functio   #
    # that you defined above.                                                #
    # Hint: You may want to use a for-loop.                                  #
    ##########################################################################
    # Replace "pass" with your code (do not modify this line)
    N, T, H = dh.shape
    _, D = cache[0][0].shape  # second element of x dimension (N, D)

    # Make null container
    dx  = torch.zeros((N, T, D)).to(dh)  # match dtype and device by torch.to()
    dh0 = torch.zeros((N, H)).to(dh) 
    dWx = torch.zeros((D, H)).to(dh) 
    dWh = torch.zeros((H, H)).to(dh) 
    db  = torch.zeros(H).to(dh) 

    # Gradient flow
    for t in range(T-1, -1, -1):
        dnext_h = dh[:, t, :]
        dx_t, dprev_h, dWx_t, dWh_t, db_t = rnn_step_backward(dh0 + dnext_h, cache[t])
        # dh0 + dnext_h
        # - is upstream gradients
        # - summation of dh0 (loss function at each timestep) 
        #   and dnext_h (gradients being passed between timesteps)
        dh0 = dprev_h  # update dprev_h
        dx[:, t, :] = dx_t  # store dx at timestep
        dWx += dWx_t  # additive
        dWh += dWh_t  # additive
        db += db_t    # additive 

    ## ref: https://github.com/AndreiKeino/EECS-498-007-598-005-Deep-Learning-for-Computer-Vision/blob/master/assignments/4/rnn_lstm_attention_captioning_completed.ipynb
    ##########################################################################
    #                            END OF YOUR CODE                            #
    ##########################################################################
    return dx, dh0, dWx, dWh, db


class RNN(nn.Module):
    """
    Single-layer vanilla RNN module.

    You don't have to implement anything here but it is highly recommended to
    read through the code as you will implement subsequent modules.
    """
    def __init__(self, input_dim: int, hidden_dim: int):
        """
        Initialize an RNN. Model parameters to initialize:
        - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
        - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
        - b: Biases, of shape (H,)

        Args:
        _ input_dim: Input size, denoted as D before
        _ hidden_dim: Hidden size, denoted as H before
        """
        super().__init__()

        # Register parameters
        self.Wx = nn.Parameter(
            torch.randn(input_dim, hidden_dim).div(math.sqrt(input_dim))
        )
        self.Wh = nn.Parameter(
            torch.randn(hidden_dim, hidden_dim).div(math.sqrt(hidden_dim))
        )
        self.b = nn.Parameter(torch.zeros(hidden_dim))

    def forward(self, x, h0):
        """
        Args:
        - x: Input data for the entire timeseries, of shape (N, T, D)
        - h0: Initial hidden state, of shape (N, H)

        Returns:
        - hn: The hidden state output
        """
        hn, _ = rnn_forward(x, h0, self.Wx, self.Wh, self.b)
        return hn

    def step_forward(self, x, prev_h):
        """
        Args:
        - x: Input data for one time step, of shape (N, D)
        - prev_h: The previous hidden state, of shape (N, H)

        Returns:
        - next_h: The next hidden state, of shape (N, H)
        """
        next_h, _ = rnn_step_forward(x, prev_h, self.Wx, self.Wh, self.b)
        return next_h


class ImageEncoder(nn.Module):
    """
    Convolutional network that accepts images as input and outputs their
    spatial grid features. This module servesx as the image encoder in image
    captioning model. We will use a tiny RegNet-X 400MF model that is
    initialized with ImageNet-pretrained weights from Torchvision library.

    NOTE: We could use any convolutional network architecture, but we opt for
    a tiny RegNet model so it can train decently with a single Tesla T4 GPU.
    """
    def __init__(self, pretrained: bool = True, verbose: bool = True):
        """
        Args:
        - pretrained: Whether to initialize this model with pretrained
          weights from Torchvision library.
        - verbose: Whether to log expected output shapes during instantiation
        """
        super().__init__()
        if pretrained:
            weights = torchvision.models.RegNet_X_400MF_Weights.IMAGENET1K_V2
        else:
            weights = None
        self.cnn = torchvision.models.regnet_x_400mf(weights=weights)

        # Torchvision models return global average pooled features by default.
        # Our attention-based models may require spatial grid features. So we
        # wrap the ConvNet with torchvision's feature extractor. We will get
        # the spatial features right before the final classification layer.
        self.backbone = feature_extraction.create_feature_extractor(
            self.cnn, return_nodes={'trunk_output.block4': 'c5'}
        )
        # We call these features "c5", a name that may sound familiar from the
        # object detection assignment. :-)

        # Pass a dummy batch of input images to infer output shape.
        dummy_out = self.backbone(torch.randn(2, 3, 224, 224))['c5']
        self._out_channels = dummy_out.shape[1]

        if verbose:
            print('For input images in NCHW format, shape (2, 3, 224, 224)')
            print(f'Shape of output c5 features: {dummy_out.shape}')

        # Input image batches are expected to be float tensors in range [0, 1].
        # However, the backbone here expects these tensors to be normalized by
        # ImageNet color mean/std (as it was trained that way).
        # We define a function to transform the input images before extraction:
        self.normalize = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

    @property
    def out_channels(self):
        """
        Number of output channels in extracted image features. You may access
        this value freely to define more modules to go with this encoder.
        """
        return self._out_channels

    def forward(self, images: torch.Tensor):
        # Input images may be uint8 tensors in [0-255], change them to float
        # tensors in [0-1]. Get float type from backbone (could be float32/64).
        if images.dtype == torch.uint8:
            images = images.to(dtype=self.cnn.stem[0].weight.dtype)
            images /= 255.0

        # Normalize images by ImageNet color mean/std.
        images = self.normalize(images)

        # Extract c5 features from encoder (backbone) and return.
        # shape: (B, out_channels, H / 32, W / 32)
        features = self.backbone(images)['c5']
        return features


class WordEmbedding(nn.Module):
    """
    Simplified version of torch.nn.Embedding.

    We operate on minibatches of size N where
    each sequence has length T. We assume a vocabulary of V words, assigning
    each word to a vector of dimension D.

    Args:
    - x: Integer array of shape (N, T) giving indices of words. Each element
      idx of x must be in the range 0 <= idx < V.

    Returns:
    - out: Array of shape (N, T, D) giving word vectors for all input words.
    """
    def __init__(self, vocab_size: int, embed_size: int):
        super().__init__()

        # Register parameters
        self.W_embed = nn.Parameter(
            torch.randn(vocab_size, embed_size).div(math.sqrt(vocab_size))
        )

    def forward(self, x):
        out = None
        ######################################################################
        # TODO: Implement the forward pass for word embeddings.              #
        ######################################################################
        # Replace "pass" with your code (do not modify this line)
        out = self.W_embed[x]  # (N, T, D)

        ## ref: No reference (My own code)
        ######################################################################
        #                          END OF YOUR CODE                          #
        ######################################################################
        return out


def temporal_softmax_loss(x, y, ignore_index=None):
    """
    A temporal version of softmax loss for use in RNNs. We assume that we are
    making predictions over a vocabulary of size V for each timestep of a
    timeseries of length T, over a minibatch of size N. The input x gives
    scores for all vocabulary elements at all timesteps, and y gives the
    indices of the ground-truth element at each timestep. We use a
    cross-entropy loss at each timestep, *summing* the loss over all timesteps
    and *averaging* across the minibatch.

    As an additional complication, we may want to ignore the model output at
    some timesteps, since sequences of different length may have been combined
    into a minibatch and padded with NULL tokens. The optional ignore_index
    argument tells us which elements in the caption should not contribute to
    the loss.

    Args:
    - x: Input scores, of shape (N, T, V)
    - y: Ground-truth indices, of shape (N, T) where each element is in the
      range 0 <= y[i, t] < V

    Returns:
    - loss: Scalar giving loss
    """
    loss = None
    ##########################################################################
    # TODO: Implement the temporal softmax loss function. Note that we       #
    # compute the cross-entropy loss at each timestep, summing the loss over #
    # all timesteps and averaging across the minibatch.                      #
    # Hint: F.cross_entropy(..., ignore_index, reduction)                    #
    ##########################################################################
    # Replace "pass" with your code (do not modify this line)
    N, _, _ = x.shape

    # to fit in F.cross_entropy (N, V, T) where V is num class fo vocabulary
    x = x.permute(0, 2, 1)  # (N, T, V) -> (N, V, T)
    # Summing
    loss = F.cross_entropy(x, y, ignore_index=ignore_index, reduction='sum')
    # Averaging
    loss /= N

    ## ref: https://github.com/AndreiKeino/EECS-498-007-598-005-Deep-Learning-for-Computer-Vision/blob/master/assignments/4/rnn_lstm_attention_captioning_completed.ipynb
    ##########################################################################
    #                            END OF YOUR CODE                            #
    ##########################################################################
    return loss


class CaptioningRNN(nn.Module):
    """
    A CaptioningRNN produces captions from images using a recurrent
    neural network.

    The RNN receives input vectors of size D, has a vocab size of V, works on
    sequences of length T, has an RNN hidden dimension of H, uses word vectors
    of dimension W, and operates on minibatches of size N.

    Note that we don't use any regularization for the CaptioningRNN.

    You will implement the `__init__` method for model initialization and
    the `forward` method first, then come back for the `sample` method later.
    """
    def __init__(
        self,
        word_to_idx,
        input_dim: int = 512,
        wordvec_dim: int = 128,
        hidden_dim: int = 128,
        cell_type: str = 'rnn',
        image_encoder_pretrained: bool = True,
        ignore_index: Optional[int] = None,
    ):
        """
        Construct a new CaptioningRNN instance.

        Args:
        - word_to_idx: A dictionary giving the vocabulary. It contains V
          entries, and maps each string to a unique integer in the range [0, V)
        - input_dim: Dimension D of input image feature vectors.
        - wordvec_dim: Dimension W of word vectors.
        - hidden_dim: Dimension H for the hidden state of the RNN.
        - cell_type: What type of RNN to use; either 'rnn' or 'lstm'.
        """
        super().__init__()
        if cell_type not in {'rnn', 'lstm', 'attn'}:
            raise ValueError("Invalid cell_type '%s'" % cell_type)

        self.cell_type = cell_type
        self.word_to_idx = word_to_idx
        self.idx_to_word = {i: w for w, i in word_to_idx.items()}

        vocab_size = len(word_to_idx)

        self._null = word_to_idx['<NULL>']
        self._start = word_to_idx.get('<START>', None)
        self._end = word_to_idx.get('<END>', None)
        self.ignore_index = ignore_index
        ######################################################################
        # TODO: Initialize the image captioning module by defining:          #
        # self.image_encoder using ImageEncoder                              #
        # self.feat_proj using nn.Linear (from CNN pooled feature to `h0`)   #
        # self.word_embed using WordEmbedding                                #
        # self.rnn using RNN, LSTM, or AttentionLSTM depending on `cell_type`#
        # self.output_proj using nn.Linear (from RNN hidden state to vocab   #
        # probability)                                                       #
        ######################################################################
        # Replace "pass" with your code (do not modify this line)
        # 1. Encode image through CNN
        self.image_encoder = ImageEncoder()

        # 2. Input for RNN
        if cell_type in {'rnn', 'lstm'}:
            self.feat_proj = nn.Sequential(
                nn.AvgPool2d(4, 4),  # Since, shape of outputs spatial features from the final layer is (B, C, H/32, W/32)
                nn.Flatten(),        # To feed into nn.Linear
                nn.Linear(input_dim, hidden_dim)
            )       
        elif cell_type == 'attn':
            self.feat_proj = nn.Conv2d(input_dim, hidden_dim, 1, stride=1)
        
        # 3. Encodding layer (Encode vocabularies to feed RNN)
        # - input (N, T, V) // output (N, T, W)
        self.word_embed = WordEmbedding(vocab_size, wordvec_dim)

        # 4. RNN 
        # - input (N, T, W) // output (N, T, H)
        if cell_type == 'rnn':
            self.rnn = RNN(wordvec_dim, hidden_dim)  # input dimension is not intput-dim
        elif cell_type == 'lstm':
            self.rnn = LSTM(wordvec_dim, hidden_dim)
        else:
            self.rnn = AttentionLSTM(wordvec_dim, hidden_dim)  # attention

        # 5. output layer
        # - input (N, T, H) // output (N, T, V)
        self.output_proj = nn.Linear(hidden_dim, vocab_size)

        ## ref: https://github.com/AndreiKeino/EECS-498-007-598-005-Deep-Learning-for-Computer-Vision/blob/master/assignments/4/rnn_lstm_attention_captioning_completed.ipynb
        ######################################################################
        #                          END OF YOUR CODE                          #
        ######################################################################

    def forward(self, images, captions):
        """
        Compute training-time loss for the RNN. We input images and the GT
        captions for those images, and use an RNN (or LSTM) to compute loss.
        The backward part will be done by torch.autograd.

        Args:
        - images: Input images, of shape (N, 3, 112, 112)
        - captions: Ground-truth captions; an integer array of shape
          (N, T + 1) where each element is in the range 0 <= y[i, t] < V

        Returns:
        - loss: A scalar loss
        """
        """
        Cut captions into two pieces: captions_in has everything but the last
        word and will be input to the RNN; captions_out has everything but the
        first word and this is what we will expect the RNN to generate. These
        are offset by one relative to each other because the RNN should produce
        word (t+1) after receiving word t. The first element of captions_in
        will be the START token, and the first element of captions_out will
        be the first word.
        """
        captions_in = captions[:, :-1]
        captions_out = captions[:, 1:]

        loss = 0.0
        ######################################################################
        # TODO: Implement the forward pass for the CaptioningRNN.            #
        # In the forward pass you will need to do the following:             #
        # 1) Apply the feature projection to project the image feature to    #
        # the initial hidden state `h0` (for RNN/LSTM, of shape (N, H); so   #
        # you need global average pooling) or the projected CNN activation   #
        # input `A` (for Attention LSTM, of shape (N, H, D_a, D_a); so you   #
        # need to permute/reshape features before and after the projection.) #
        # 2) Use a word embedding layer to transform words in captions_in    #
        # from indices to vectors, giving an array of shape (N, T, W).       #
        # 3) Use either a vanilla RNN or LSTM (depending on self.cell_type)  #
        # to process the sequence of input word vectors and produce hidden   #
        # state vectors for all timesteps, producing an array of shape       #
        # (N, T, H).                                                         #
        # 4) Apply the output projection to compute scores over the          #
        # vocabulary at every timestep from the hidden states, giving an     #
        # array of shape (N, T, V).                                          #
        # 5) Use (temporal) softmax to compute loss using captions_out,      #
        # ignoring the points where the output word is <NULL>.               #
        # Do not worry about regularizing the weights or their gradients!    #
        ######################################################################
        # Replace "pass" with your code (do not modify this line)
        # 1. Feature projection
        features = self.image_encoder(images)   # feature extraction of the images; (N, C, H, W)
        h0 = self.feat_proj(features)           # project the image feature to the initial hidden state 'ho'; (N, C, W, H) -> (N, H)
        
        # 2. Word embedding layer
        embedded = self.word_embed(captions_in)  # Embed the vocab at each timestep; (N, T) -> (N, T, W)
        
        # 3. RNN or LSTM
        hidden_state = self.rnn(embedded, h0)    # Apply RNN model; (N, T, W) -> (N, T, H)
        
        # 4. Compute scores over the vocabulary at each timestep
        voc = self.output_proj(hidden_state)     # Project to align dimension with V; (N, T, H) -> (N, T, V)
        
        # 5. Calculate loss with with captions_out
        loss = temporal_softmax_loss(voc, captions_out, self.ignore_index)

        ## ref: https://github.com/AndreiKeino/EECS-498-007-598-005-Deep-Learning-for-Computer-Vision/blob/master/assignments/4/rnn_lstm_attention_captioning_completed.ipynb
        ######################################################################
        #                          END OF YOUR CODE                          #
        ######################################################################
        return loss

    def sample(self, images, max_length=16):
        """
        Run a test-time forward pass for the model, sampling captions for input
        feature vectors.

        At each timestep, we embed the current word, pass it and the previous
        hidden state to the RNN to get the next hidden state, use the hidden
        state to get scores for all vocab words, and choose the word with the
        highest score as the next word. The initial hidden state is computed by
        applying an affine transform to the image features, and the initial
        word is the <START> token.

        For LSTMs you will also have to keep track of the cell state; in that
        case the initial cell state should be zero.

        Args:
        - images: Input images, of shape (N, 3, 112, 112)
        - max_length: Maximum length T of generated captions

        Returns a tuple of:
        - captions: Array of shape (N, max_length) giving sampled captions,
          where each element is an integer in the range [0, V).
          The first element (captions[:, 0]) should be the <START> token.
        - attn_weights_all: Array of shape (N, max_length, D_a, D_a) giving
          attention weights returned only when self.cell_type == 'attn'
          The first attn weights (attn_weights_all[:, 0]) should be all zero.
        """
        N = images.shape[0]
        captions = torch.full((N, max_length), self._null,
                              dtype=torch.long, device=images.device)

        if self.cell_type == 'attn':
            D_a = 4
            attn_weights_all = \
                torch.zeros(N, max_length, D_a, D_a,
                            dtype=torch.float, device=images.device)
        ######################################################################
        # TODO: Implement test-time sampling for the model. You need to      #
        # initialize the hidden state `h0` by applying self.feat_proj to the #
        # image features. For LSTM, as we provided in LSTM forward function, #
        # you need to set the initial cell state `c0` to zero.               #
        # For AttentionLSTM, `c0 = h0`. The first word that you feed to      #
        # the RNN should be the <START> token; its value is stored in the    #
        # variable self._start. After initial setting, at each timestep, you #
        # will need to do to:                                                #
        # 1) Embed the previous word using the learned word embeddings.      #
        # 2) Make an RNN step using the previous hidden state and the        #
        # embedded current word to get the next hidden state.                #
        # 3) Apply the output projection to the next hidden state            #
        # to get scores for all words in the vocabulary.                     #
        # 4) Select the word with the highest score as the next word,        #
        # writing it (the word index) to the appropriate slot in the         #
        # captions variable.                                                 #
        # For simplicity, you do not need to stop generating after an <END>  #
        # token is sampled, but you can do so if you want.                   #
        # Hint: We are working over minibatches in this function.            #
        ######################################################################
        # Replace "pass" with your code (do not modify this line)

        # 0-1. Extract features from input images
        features = self.image_encoder(images)  # (N, 3, 4, 4)

        # 0-2. Initialize setting
        # first word to the RNN
        prev_words = torch.full((N, ), self._start)
        captions[:, 0] = prev_words
        # Initial hidden state h0 and cell state c0
        if self.cell_type == 'rnn':
            prev_h = self.feat_proj(features)
        elif self.cell_type == 'lstm':
            prev_h = self.feat_proj(features)
            prev_c = torch.zeros_like(prev_h)  # lstm need cell state filled with zeros
        elif self.cell_type == 'attn':
            Attn = self.feat_proj(features)
            prev_h = Attn.mean(dim=(2, 3))  # apply mean to make prev_h
            prev_c = prev_h.clone()
        
        # Run the RNN at each timestep
        for t in range(max_length-1):

            # 1. Embed the previous word
            embedded = self.word_embed(prev_words)  # (N, H)
            # 2. RNN step: Get the next hidden state using embedded voc and previous hidden state
            if self.cell_type == 'rnn':
                # next hidden state
                prev_h = self.rnn.step_forward(embedded, prev_h)  # (N, H)
            elif self.cell_type == 'lstm':
                prov_h, prev_c = self.rnn.step_forward(embedded, prev_h, prev_c)  # (N, H)
            elif self.cell_type == 'attn':
                attn, attn_weights = dot_product_attention(prev_h, Attn) 
                attn_weights_all[:, t] = attn_weights
                prev_h, prev_c = self.rnn.step_forward(embedded, prev_h, prev_c, attn) # (N, H)

            # 3. Get scores for all words in the vocabulary (Apply output projection)
            voc = self.output_proj(prev_h)  # (N, V)

            # 4. Select hightest scores as next word
            prev_words = torch.argmax(voc, axis=1)  # (N, )
            captions[:, t+1] = prev_words   # update prev_word

        ## ref: https://github.com/AndreiKeino/EECS-498-007-598-005-Deep-Learning-for-Computer-Vision/blob/master/assignments/4/rnn_lstm_attention_captioning_completed.ipynb
        ######################################################################
        #                          END OF YOUR CODE                          #
        ######################################################################
        if self.cell_type == 'attn':
            return captions, attn_weights_all.cpu()
        else:
            return captions


class LSTM(nn.Module):
    """
    Single-layer, uni-directional LSTM module.
    """
    def __init__(self, input_dim: int, hidden_dim: int):
        """
        Initialize a LSTM. Model parameters to initialize:
        - Wx: Weights for input-to-hidden connections, of shape (D, 4H)
        - Wh: Weights for hidden-to-hidden connections, of shape (H, 4H)
        - b: Biases, of shape (4H,)

        Args:
        - input_dim: Input size, denoted as D before
        - hidden_dim: Hidden size, denoted as H before
        """
        super().__init__()

        # Register parameters
        self.Wx = nn.Parameter(
            torch.randn(input_dim, hidden_dim * 4).div(math.sqrt(input_dim))
        )
        self.Wh = nn.Parameter(
            torch.randn(hidden_dim, hidden_dim * 4).div(math.sqrt(hidden_dim))
        )
        self.b = nn.Parameter(torch.zeros(hidden_dim * 4))

    def step_forward(
        self, x: torch.Tensor, prev_h: torch.Tensor, prev_c: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for a single timestep of an LSTM.
        The input data has dimension D, the hidden state has dimension H, and
        we use a minibatch size of N.

        Args:
        - x: Input data for one time step, of shape (N, D)
        - prev_h: The previous hidden state, of shape (N, H)
        - prev_c: The previous cell state, of shape (N, H)
        - Wx: Input-to-hidden weights, of shape (D, 4H)
        - Wh: Hidden-to-hidden weights, of shape (H, 4H)
        - b: Biases, of shape (4H,)

        Returns a tuple of:
        - next_h: Next hidden state, of shape (N, H)
        - next_c: Next cell state, of shape (N, H)
        """
        next_h, next_c = None, None
        ######################################################################
        # TODO: Implement the forward pass for a single timestep of an LSTM. #
        # Hint: torch.sigmoid, torch.tanh                                    #
        ######################################################################
        # Replace "pass" with your code (do not modify this line)
        _, H = prev_h.shape
        # Compute an activation vector (N, 4H)
        activation = torch.mm(x, self.Wx) + torch.mm(prev_h, self.Wh) + self.b

        # Divide into four vector; each vector is (N, H)
        input_gate = torch.sigmoid(activation[:, : H*1])
        forget_gate = torch.sigmoid(activation[:, H*1 : H*2])
        output_gate = torch.sigmoid(activation[:, H*2 : H*3])
        gate_gate = torch.tanh(activation[:, H*3 : H*4])

        # Compute next cell state and next hidden state
        next_c = forget_gate * prev_c + input_gate * gate_gate
        next_h = output_gate * torch.tanh(next_c)

        ## ref: No reference (My own code)
        ######################################################################
        #                          END OF YOUR CODE                          #
        ######################################################################
        return next_h, next_c

    def forward(self, x: torch.Tensor, h0: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for an LSTM over an entire sequence of data. We assume an
        input sequence composed of T vectors, each of dimension D. The LSTM
        uses a hidden size of H, and we work over a minibatch containing N
        sequences. After running the LSTM forward, we return the hidden states
        for all timesteps.

        Note that the initial cell state is set to zero, and the final cell
        state is not returned; it is an internal variable to the LSTM and is
        not accessed from outside.

        Args:
        - x: Input data for the entire timeseries, of shape (N, T, D)
        - h0: Initial hidden state, of shape (N, H)

        Returns:
        - hn: The hidden state output.
        """
        c0 = torch.zeros_like(h0)  # initial cell state
        hn = None
        ######################################################################
        # TODO: Implement the forward pass for an LSTM running on a sequence #
        # of input data.                                                     #
        ######################################################################
        # Replace "pass" with your code (do not modify this line)
        N, T, D = x.shape
        H = h0.shape[1]
        hn = torch.zeros(N, T, H).to(h0)  # Container for hidden state for all minibatch

        prev_h = h0.clone()
        prev_c = c0.clone()
        for t in range(T):
            xt = x[:, t, :]
            prev_h, prev_c = self.step_forward(xt, prev_h, prev_c)
            hn[:, t, :] = prev_h

        ## ref: No reference (My own code)
        ######################################################################
        #                          END OF YOUR CODE                          #
        ######################################################################
        return hn


def dot_product_attention(prev_h, A):
    """
    A simple scaled dot-product attention layer.

    Args:
    - prev_h: The LSTM hidden state from previous time step, of shape (N, H)
    - A: **Projected** CNN feature activation, of shape (N, H, D_a, D_a),
       where H is the LSTM hidden state size

    Returns a tuple of:
    - attn: Attention embedding output, of shape (N, H)
    - attn_weights: Attention weights, of shape (N, D_a, D_a)

    """
    N, H, D_a, _ = A.shape

    attn, attn_weights = None, None
    ##########################################################################
    # TODO: Implement the scaled dot-product attention we described earlier. #
    # HINT: torch.bmm, torch.softmax                                         #
    # Make sure you reshape `attn_weights` back to (N, D_a, D_a).            #
    ##########################################################################
    # Replace "pass" with your code (do not modify this line)
    # 1. Flatten the spatial dimension of A
    A = torch.flatten(A, start_dim=2)  # (N, H, D_a, D_a) -> (N, H, D_a**2)

    # 2. Calculate alignment score
    # Calculate alignment score with previous hidden state and input
    H = torch.as_tensor(H, dtype=prev_h.dtype, device=prev_h.device)  # To use torch.sqrt()
    attn_alignment = torch.bmm(torch.unsqueeze(prev_h, dim=1), A) / torch.sqrt(H)  # (N, 1, D_a ** 2)
    attn_alignment = attn_alignment.permute(0, 2, 1)       # (N, D_a ** 2, 1)

    # 3. Get attention weights
    attn_weights = torch.softmax(attn_alignment, dim=1) # summation along 2rd axis == 1

    # 5. Get attention embedding output
    attn = torch.bmm(A, attn_weights).reshape(N, -1)  # by scaled dot-product attention
    attn_weights = attn_weights.reshape(N, D_a, D_a)

    ## ref: https://github.com/AndreiKeino/EECS-498-007-598-005-Deep-Learning-for-Computer-Vision/blob/master/assignments/4/rnn_lstm_attention_captioning_completed.ipynb
    ##########################################################################
    #                            END OF YOUR CODE                            #
    ##########################################################################
    return attn, attn_weights


class AttentionLSTM(nn.Module):
    """
    This is our single-layer, uni-directional Attention module.

    Args:
    - input_dim: Input size, denoted as D before
    - hidden_dim: Hidden size, denoted as H before
    """
    def __init__(self, input_dim: int, hidden_dim: int):
        """
        Initialize a LSTM. Model parameters to initialize:
        - Wx: Weights for input-to-hidden connections, of shape (D, 4H)
        - Wh: Weights for hidden-to-hidden connections, of shape (H, 4H)
        - Wattn: Weights for attention-to-hidden connections, of shape (H, 4H)
        - b: Biases, of shape (4H,)
        """
        super().__init__()

        # Register parameters
        self.Wx = nn.Parameter(
            torch.randn(input_dim, hidden_dim * 4).div(math.sqrt(input_dim))
        )
        self.Wh = nn.Parameter(
            torch.randn(hidden_dim, hidden_dim * 4).div(math.sqrt(hidden_dim))
        )
        self.Wattn = nn.Parameter(
            torch.randn(hidden_dim, hidden_dim * 4).div(math.sqrt(hidden_dim))
        )
        self.b = nn.Parameter(torch.zeros(hidden_dim * 4))

    def step_forward(
        self,
        x: torch.Tensor,
        prev_h: torch.Tensor,
        prev_c: torch.Tensor,
        attn: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
        - x: Input data for one time step, of shape (N, D)
        - prev_h: The previous hidden state, of shape (N, H)
        - prev_c: The previous cell state, of shape (N, H)
        - attn: The attention embedding, of shape (N, H)

        Returns:
        - next_h: The next hidden state, of shape (N, H)
        - next_c: The next cell state, of shape (N, H)
        """
        next_h, next_c = None, None
        ######################################################################
        # TODO: Implement the forward pass for a single timestep of an       #
        # attention LSTM, which should be very similar to LSTM.step_forward  #
        ######################################################################
        # Replace "pass" with your code (do not modify this line)
        _, H = prev_h.shape
        # Compute an activation vector (N, 4H)
        activation = torch.mm(x, self.Wx) + torch.mm(prev_h, self.Wh) + torch.mm(attn, self.Wattn) + self.b

        # Divide into four vector; each vector is (N, H)
        input_gate = torch.sigmoid(activation[:, : H*1])
        forget_gate = torch.sigmoid(activation[:, H*1 : H*2])
        output_gate = torch.sigmoid(activation[:, H*2 : H*3])
        gate_gate = torch.tanh(activation[:, H*3 : H*4])

        # Compute next cell state and next hidden state
        next_c = forget_gate * prev_c + input_gate * gate_gate
        next_h = output_gate * torch.tanh(next_c)

        ## ref: No reference (My own code)
        ######################################################################
        #                          END OF YOUR CODE                          #
        ######################################################################
        return next_h, next_c

    def forward(self, x: torch.Tensor, A: torch.Tensor):
        """
        Forward pass for an LSTM over an entire sequence of data. We assume an
        input sequence composed of T vectors, each of dimension D. The LSTM
        uses a hidden size of H, and we work over a minibatch containing
        N sequences. After running the LSTM forward, we return hidden states
        for all timesteps.

        Note that the initial cell state is passed as input, but the initial
        cell state is set to zero. Also note that the cell state is not
        returned; it is an internal variable to the LSTM and is not accessed
        from outside.

        h0 and c0 are same initialized as global image feature (avgpooled A)
        For simplicity, we implement scaled dot-product attention, which means
        in Eq. 4 of the paper (https://arxiv.org/pdf/1502.03044.pdf),
        f_{att}(a_i, h_{t-1}) equals to the scaled dot product of
        a_i and h_{t-1}.

        Args:
        - x: Input data for the entire timeseries, of shape (N, T, D)
        - A: The projected CNN feature activation, of shape (N, H, 4, 4)

        Returns:
        - hn: The hidden state output
        """

        # The initial hidden state h0 and cell state c0 are initialized
        # differently in AttentionLSTM from the original LSTM and hence
        # we provided them for you.
        h0 = A.mean(dim=(2, 3))  # Initial hidden state, of shape (N, H)
        c0 = h0  # Initial cell state, of shape (N, H)
        hn = None
        ######################################################################
        # TODO: Implement the forward pass for an attention LSTM running on  #
        # a sequence of input data, which should be very similar to          #
        # LSTM.forward                                                       #
        # Hint: dot_product_attention                                        #
        ######################################################################
        # Replace "pass" with your code (do not modify this line)
        N, T, D = x.shape
        H = h0.shape[1]
        hn = torch.zeros(N, T, H).to(h0)  # Container for hidden state for all minibatch

        prev_h = h0.clone()
        prev_c = c0.clone()
        for t in range(T):
            xt = x[:, t, :]
            attn, _ = dot_product_attention(prev_h, A)
            prev_h, prev_c = self.step_forward(xt, prev_h, prev_c, attn)
            hn[:, t, :] = prev_h

        ## ref: No reference (My own code)
        ######################################################################
        #                          END OF YOUR CODE                          #
        ######################################################################
        return hn
