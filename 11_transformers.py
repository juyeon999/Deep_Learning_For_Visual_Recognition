"""
Implements a Transformer in PyTorch.
WARNING: you SHOULD NOT use ".to()" or ".cuda()" in each implementation block.
"""
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F


def hello():
    """
    This is a sample function that we will try to import and run to ensure
    that our environment is correctly set up on Google Colab.
    """
    print("Hello from transformers.py!")


def generate_token_dict(vocab):
    """
    The function creates a hash map from the elements in the vocabulary to
    to a unique positive integer value.

    Args:
    - vocab: A list of strings containing all the items in the vocabulary

    Returns:
    - token_dict: A python dictionary with key as a string item in the vocab
      and value as a unique integer value
    """
    # initialize an empty dictionary
    token_dict = {}
    ##########################################################################
    # TODO: Complete the dictionary `token_dict`, where each key is an       #
    # element in the list `vocab` and the corresponding value is a unique    #
    # index. Specifically, map the first element in `vocab` to 0, the last   #
    # element in `vocab` to len(vocab)-1, and the elements in between to     #
    # consecutive numbers in between 0 and len(vocab)-1.                     #
    # Hint: python built-in `enumerate`                                      #
    ##########################################################################
    # Replace "pass" with your code (do not modify this line)
    for i, voc in enumerate(vocab):
        token_dict[voc] = i
    
    ## ref: No reference (my own code)
    ##########################################################################
    #                            END OF YOUR CODE                            #
    ##########################################################################
    return token_dict


def str2token(
    input_str: str, token_dict: dict, spc_tokens: list
) -> list:
    """
    The function converts an input string into a list of non-negative integers
    that will enable us to process the string using neural nets later, where
    token_dict gives the mapping from the input string to an integer.
    Keep in mind that we assign a value for each integer present in the input
    sequence. For example, for a number present in the input sequence '33',
    you should break it down to a list of digits ['0', '3'] and assign it to a
    corresponding value in token_dict.

    Args:
    - input_str: A single string in the input data, e.g.,
      "BOS POSITIVE 0333 add POSITIVE 0696 EOS"
    - token_dict: A dictionary of tokens, having a key as a string and
      a corresponding value as a unique non-negative integer
    - spc_tokens: The special tokens apart from digits
    Returns:
    - out_tokens: A list of integers corresponding to the input string
    """
    out = []
    ##########################################################################
    # TODO: For each number present in the input sequence, break it down     #
    # into a list of digits, and use the list to assign an appropriate value #
    # from token_dict. For special tokens present in the input string,       #
    # assign an appropriate value for the complete token.                    #
    ##########################################################################
    # Replace "pass" with your code (do not modify this line)
    for word in input_str.split():
        if word.isdigit():
            for digit in word:
                out.append(token_dict[digit])
        else:
            out.append(token_dict[word])

    ## ref: no reference (my own code)
    ##########################################################################
    #                            END OF YOUR CODE                            #
    ##########################################################################
    return out


def pos_enc_simple(seq_len: int, emb_dim: int) -> Tensor:
    """
    Simple positional encodings using uniform intervals for a sequence.

        PE_p = p/seq_len,

    where p is a 0-base sequence index.

    Args:
    - seq_len: An integer giving the sequence length
    - emb_dim: An integer giving the embedding dimension for the sequence

    Returns:
    - out: A Tensor of shape (1, seq_len, emb_dim) giving positional encodings
    """
    out = None
    ##########################################################################
    # TODO: Given the length of input sequence seq_len, construct a Tensor   #
    # of length seq_len with p-th element as p/seq_len, where p starts from  #
    # 0. Replicate it emb_dim times to create a tensor of the output shape.  #
    # Hint: torch.arange, torch.repeat                                       #
    ##########################################################################
    # Replace "pass" with your code (do not modify this line)
    out = torch.arange(0, 1, 1/seq_len).repeat(emb_dim, 1).T

    ## ref: no reference (my own code)
    ##########################################################################
    #                            END OF YOUR CODE                            #
    ##########################################################################
    return out


def pos_enc_sinusoid(seq_len: int, emb_dim: int) -> Tensor:
    """
    Sinusoidal positional encodings.

        PE_(p,2i)   = sin(omega_i p),
        PE_(p,2i+1) = cos(omega_i p),
        omega_i = 1/10000^{2i/emb_dim},

    where p is a 0-base sequence index, i is a 0-base embedding dim index.

    Args:
    - seq_len: An integer giving sequence length
    - emb_dim: An integer giving embedding dimension for the sequence

    Returns:
    - out: A Tensor of shape (1, seq_len, emb_dim) giving positional encodings
    """
    out = None
    ##########################################################################
    # TODO: Given the length of input sequence seq_len and embedding         #
    # dimension emb_dim, construct a Tensor of shape (seq_len, emb_dim)      #
    # where the values are from sinusoidal waves. Make sure alternating sine #
    # and cosine waves along with the embedding dimension.                   #
    # Hint: torch.arange, torch.where                                        #
    ##########################################################################
    # Replace "pass" with your code (do not modify this line)
    out = torch.zeros((seq_len, emb_dim))
    i = torch.arange(0, emb_dim // 2, 1)

    for p in range(seq_len):
        w = 10000**(- 2 * i / emb_dim)
        out[p, 2 * i] = torch.sin(p * w)
        out[p, 2 * i + 1] = torch.cos(p * w)
        if emb_dim % 2 != 0:
            w = 10000**(- 2 * (i[-1] + 1) / emb_dim)
            out[p, 2 * (i[-1] + 1)] = torch.sin(p * w)
    
    ## ref: no reference
    ##########################################################################
    #                            END OF YOUR CODE                            #
    ##########################################################################
    return out


def scaled_dot_product_two_loop_single(
    query: Tensor, key: Tensor, value: Tensor
) -> Tensor:
    """
    The function computes the scaled dot product for attention using
    two for-loops given queries Q, keys K, and values V.

    Attn(Q, K, V) = softmax(Q K^T / sqrt(emb_dim)) V

    Args:
    - query: A Tensor of shape (seq_len, emb_dim) giving the queries
    - key: A Tensor of shape (seq_len, emb_dim) giving the keys
    - value: A Tensor of shape (seq_len, emb_dim) giving the values

    Returns:
    - out: A Tensor of shape (seq_len, emb_dim) giving the scaled dot-product
      attention
    """
    out = None
    ##########################################################################
    # TODO: Implement scaled dot-product attention using two for-loops.      #
    # Specifically, you can follow the steps below:                          #
    # 1. For each query, compute dot product with each key in the sequence.  #
    # 2. Scale the scalar output by a factor of 1/sqrt(emb_dim).             #
    # 3. Apply softmax over the (seq_len) affinity scores for each query.    #
    # 4. Apply matrix-vector multiplication to compute weighted sum of       #
    # values, where weights are normalized affinity scores.                  #
    ##########################################################################
    # Replace "pass" with your code (do not modify this line)
    seq_len, emb_dim = query.shape
    out = torch.zeros_like(query)

    # For each query
    for i in range(seq_len):
        # 1. Compute dot product and scaled
        scores = torch.matmul(query[i], key.transpose(0, 1)) / (emb_dim ** 0.5)

        # 2. Apply softmax over the scores for each query
        scores = torch.softmax(scores, dim=0)  # dim=0; for each query

        # 3. Apply matrix-vector multiplication for weighted sum
        for j in range(seq_len):
            out[i] += scores[j] * value[j]  # weighted sum
        
    ## ref: ChatGPT and modify it.
    ## queries: 
    # TODO: Implement scaled dot-product attention using two for-loops.      #
    # Specifically, you can follow the steps below:                          #
    # 1. For each query, compute dot product with each key in the sequence.  #
    # 2. Scale the scalar output by a factor of 1/sqrt(emb_dim).             #
    # 3. Apply softmax over the (seq_len) affinity scores for each query.    #
    # 4. Apply matrix-vector multiplication to compute weighted sum of       #
    # values, where weights are normalized affinity scores.                  #
    ##########################################################################
    #                            END OF YOUR CODE                            #
    ##########################################################################
    return out


def scaled_dot_product_two_loop_batch(
    query: Tensor, key: Tensor, value: Tensor
) -> Tensor:
    """
    The function computes the scaled dot product for attention using
    two for-loops given batch of queries Q, keys K, and values V.

    Attn(Q, K, V) = softmax(Q K^T / sqrt(emb_dim)) V

    Args:
    - query: A Tensor of shape (batch_size, seq_len, emb_dim) giving
      the queries
    - key: A Tensor of shape (batch_size, seq_len, emb_dim) giving the keys
    - value: A Tensor of shape (batch_size, seq_len, emb_dim) giving the values

    Returns:
    - out: A Tensor of shape (batch_size, seq_len, emb_dim) giving the scaled
      dot-product attention
    """
    batch_size, seq_len, emb_dim = query.shape

    out = None
    ##########################################################################
    # TODO: Implement scaled dot-product attention for a batch of inputs     #
    # using two for-loops. Specifically, you can follow the steps below:     #
    # 1. For each query, compute dot product with each key in the sequence,  #
    # resulting a vector of length batch_size.                               #
    # 2. Scale the scalar output by a factor of 1/sqrt(emb_dim).             #
    # 3. Apply softmax over the (seq_len) affinity scores for each query.    #
    # 4. Apply batch matrix-matrix multiplication to compute weighted sum of #
    # values, where weights are normalized affinity scores.                  #
    # Hint: torch.bmm                                                        #
    ##########################################################################
    # Replace "pass" with your code (do not modify this line)
    affinity_mat = torch.zeros((batch_size, seq_len, seq_len), dtype=query.dtype, device=query.device)
    for j in range(batch_size):
        scores = torch.zeros((seq_len, seq_len), dtype=query.dtype, device=query.device)

        # 1. For each query
        for i in range(seq_len):
            # 1. Compute dot product and scaled
            score = torch.matmul(query[j, i, :], key[j].transpose(0, 1)) / (emb_dim ** 0.5)
        
            # 2. Apply softmax over the scores for each query
            scores[i] = torch.softmax(score, dim=0)  # dim=0; for each query
        # save as matrix
        affinity_mat[j] = scores
    
    # 3. Apply batch matrix-matrix multiplication to compute weighted sum of values
    out = torch.bmm(affinity_mat, value)

    ## ref: no reference (My own code)
    ##########################################################################
    #                            END OF YOUR CODE                            #
    ##########################################################################
    return out


def scaled_dot_product_no_loop_batch(
    query: Tensor, key: Tensor, value: Tensor, mask: Tensor = None,
    inf: float = 1e9
) -> Tensor:
    """
    The function computes the scaled dot product for attention using
    two for-loops given batch of queries Q, keys K, and values V.

    Attn(Q, K, V) = softmax(Q K^T / sqrt(emb_dim)) V

    Args:
    - query: A Tensor of shape (batch_size, seq_len, emb_dim) giving
      the queries
    - key: A Tensor of shape (batch_size, seq_len, emb_dim) giving the keys
    - value: A Tensor of shape (batch_size, seq_len, emb_dim) giving the values
    - mask: A bool Tensor of shape (batch_size, seq_len, seq_len) giving the
      mask applied to the affinity matrix for scaled dot-product attention

    Returns:
    - out: A Tensor of shape (batch_size, seq_len, emb_dim) giving the scaled
      dot-product attention
    - affinity_mat: A Tensor of shape (batch_size, seq_len, seq_len) giving the
      softmax affinity matrix
    """
    batch_size, seq_len, emb_dim = query.shape

    out, affinity_mat = None, None
    ##########################################################################
    # TODO: Implement scaled dot-product attention for a batch of inputs     #
    # **without** for-loops. Specifically, you can follow the steps below:   #
    # 1. Compute dot product between queries and keys, resulting a Tensor of #
    # shape (batch_size, seq_len, seq_len).                                  #
    # 2. Scale the output by a factor of 1/sqrt(emb_dim).                    #
    # 3. Apply softmax over the (seq_len) affinity scores for each query.    #
    # 4. Apply batch matrix-matrix multiplication to compute weighted sum of #
    # values, where weights are normalized affinity scores.                  #
    #                                                                        #
    # If you come here from the Transformer Decoder section:                 #
    # If mask is not None, apply the mask to the (unnormalized) affinity     #
    # matrix by assigning -inf (given as an argument) to the positions where #
    # the mask value is True.                                                #
    #                                                                        #
    # Hint: torch.bmm, torch.masked_fill                                     #
    ##########################################################################
    # Replace "pass" with your code (do not modify this line)
    # 1~2. Compute dot product between queries and keys and scaled
    scores = torch.bmm(query, key.transpose(1, 2)) / (emb_dim ** 0.5)  # Q K^T and scale

    # Mask in inference time (we inference sequentially so we do not know the next token)
    # It should be masked before softmax function, -inf -> 0 by softmax
    if mask is not None:
        scores = scores.masked_fill(mask, -inf)  # if mask is True, fill -inf or just scores

    # 3. Apply softmax over the (seq_len) affinity scores for each query
    affinity_mat = torch.softmax(scores, dim=-1)  # dim=-1 gives softmax over each query

    # 4. Apply batch matrix-matrix multiplication to compute weighted sum of values
    out = torch.bmm(affinity_mat, value)  # matrix-matrix multiplication for each batch

    ## ref: ChatGPT
    ## query:
    # TODO: Implement scaled dot-product attention for a batch of inputs     #
    # **without** for-loops. Specifically, you can follow the steps below:   #
    # 1. Compute dot product between queries and keys, resulting a Tensor of #
    # shape (batch_size, seq_len, seq_len).                                  #
    # 2. Scale the output by a factor of 1/sqrt(emb_dim).                    #
    # 3. Apply softmax over the (seq_len) affinity scores for each query.    #
    # 4. Apply batch matrix-matrix multiplication to compute weighted sum of #
    # values, where weights are normalized affinity scores.                  #
    #                                                                        #
    # If you come here from the Transformer Decoder section:                 #
    # If mask is not None, apply the mask to the (unnormalized) affinity     #
    # matrix by assigning -inf (given as an argument) to the positions where #
    # the mask value is True.                                                #
    #                                                                        #
    # Hint: torch.bmm, torch.masked_fill                                     #
    ##########################################################################
    #                            END OF YOUR CODE                            #
    ##########################################################################
    return out, affinity_mat


class SingleHeadAttention(nn.Module):
    def __init__(self, dim_in: int, dim_q: int, dim_v: int):
        """
        This class implements a single-head attention layer.
        It processes input queries, keys, and values using Linear layers and
        apply scaled_dot_product_no_loop_batch to compute the output.

        Args:
            dim_in: An integer giving the input sequence embedding dimension
            dim_q: An integer giving the dimension of query and key vectors
            dim_v: An integer giving the dimension of value vectors
        """
        super().__init__()

        self.q = None  # for query
        self.k = None  # for key
        self.v = None  # for value
        self.affinity_mat = None
        ######################################################################
        # TODO: Initialize Linear functions transforming the input queries,  #
        # keys, and values from dim_in to dim_q, dim_q, and dim_v dimension, #
        # respectively. Apply the following initialization strategy:         #
        # For a Linear layer mapping from D_in to D_out dimension, sample    #
        # weights from a uniform distribution bounded by [-c, c], where      #
        # c = sqrt(6/(D_in + D_out))                                         #
        # Hint: nn.init.uniform_                                             #
        ######################################################################
        # Replace "pass" with your code (do not modify this line)
        ## 1. Linear Transformation
        self.q = nn.Linear(dim_in, dim_q)
        self.k = nn.Linear(dim_in, dim_q)
        self.v = nn.Linear(dim_in, dim_v)

        ## 2. Initialize the weights
        c = (6 / (dim_in + dim_q)) ** (0.5)  # for self.q and self.k
        nn.init.uniform_(self.q.weight.data, -c, c)
        nn.init.uniform_(self.k.weight.data, -c, c)

        c = (6 / (dim_in + dim_v)) ** (0.5)
        nn.init.uniform_(self.v.weight.data, -c, c)

        # ref: No reference (my own code)
        ######################################################################
        #                          END OF YOUR CODE                          #
        ######################################################################

    def forward(
        self, query: Tensor, key: Tensor, value: Tensor, mask: Tensor = None
    ) -> Tensor:
        """
        The forward pass for a single-head attention layer.

        Args:
        - query: A Tensor of shape (batch_size, seq_len, emb_dim) giving
          the queries
        - key: A Tensor of shape (batch_size, seq_len, emb_dim) giving the keys
        - value: A Tensor of shape (batch_size, seq_len, emb_dim) giving
          the values
        - mask: A bool Tensor of shape (batch_size, seq_len, seq_len) giving
          the mask applied to the affinity matrix for scaled dot-product
          attention

        Returns:
        - out: A Tensor of shape (batch_size, seq_len, emb_dim) giving
          the attention output
        """
        out = None
        ######################################################################
        # TODO: Implement the single-head attention forward pass.            #
        # Transform input queries, keys, and values using Linear layers and  #
        # pass them with mask to scaled_dot_product_no_loop_batch. Store the #
        # affinity matrix in self.affinity_mat and return the output.        #
        ######################################################################
        # Replace "pass" with your code (do not modify this line)
        query = self.q(query)
        key = self.k(key)
        value = self.v(value)
        out, self.affinity_mat = scaled_dot_product_no_loop_batch(query, key, value, mask)
        
        # ref: No reference (my own code)
        ######################################################################
        #                          END OF YOUR CODE                          #
        ######################################################################
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads: int, dim_in: int, dim_out: int):
        """
        A naive implementation of a multi-head attention layer.
        The heads are defined as a nn.ModuleList, where each head is a
        SingleHeadAttention layer.
        The input is processed in the heads in parallel, and then concatenated.

        Args:
        - num_heads: An integer giving the number of heads
        - dim_in: An integer giving the dimension of the input queries, keys,
          and values
        - dim_out: An integer giving the dimension of the concatenation of the
          output of the SingleHeadAttention layers
        """
        super().__init__()

        assert dim_out % num_heads == 0, \
            'dim_out should be divisible by num_heads'

        self.heads = None  # nn.ModuleList of heads
        self.proj = None  # from dim_out to dim_in
        ######################################################################
        # TODO: Initialize the heads and projection layer. The heads are     #
        # nn.ModuleList of SingleHeadAttention layers, each of which gets    #
        # dim_in dimensional inputs and returns (dim_out / num_heads)        #
        # dimensional outputs. The projection layer is a Linear layer        #
        # mapping the dimensionality back to dim_in, initialized by the same #
        # strategy as described in SingleHeadAttention.                      #
        ######################################################################
        # Replace "pass" with your code (do not modify this line)
        self.heads = nn.ModuleList(
            SingleHeadAttention(dim_in, dim_out // num_heads, dim_out // num_heads)
            for i in range(num_heads)  
            # pytorch documents for nn.ModuleList: https://pytorch.org/docs/stable/generated/torch.nn.ModuleList.html
        )

        self.proj = nn.Linear(dim_out, dim_in)  # linear mapping 
        c = (6 / (dim_in + dim_out)) ** (0.5)
        nn.init.uniform_(self.proj.weight.data, -c, c)  # initialize

        ## ref: No reference. (Only refer pytorch official documents and above code)
        ######################################################################
        #                          END OF YOUR CODE                          #
        ######################################################################

    def forward(
        self, query: Tensor, key: Tensor, value: Tensor, mask: Tensor = None
    ) -> Tensor:
        """
        The forward pass for a multi-head attention layer.

        Args:
        - query: A Tensor of shape (batch_size, seq_len, emb_dim) giving
          the queries
        - key: A Tensor of shape (batch_size, seq_len, emb_dim) giving the keys
        - value: A Tensor of shape (batch_size, seq_len, emb_dim) giving
          the values
        - mask: A bool Tensor of shape (batch_size, seq_len, seq_len) giving
          the mask applied to the affinity matrix for scaled dot-product
          attention

        Returns:
        - out: A Tensor of shape (batch_size, seq_len, emb_dim) giving
          the attention output
        """
        out = None
        ######################################################################
        # TODO: Implement the multi-head attention forward pass.             #
        # Pass inputs to all heads, concatenate outputs of heads, and then   #
        # feed it to the projection layer to get the output tensor of the    #
        # same shape as the input.                                           #
        ######################################################################
        # Replace "pass" with your code (do not modify this line)
        # 1. Pass inputs to all heads
        head_outputs = []
        for head in self.heads:
            head_output = head(query, key, value, mask)  # pop the Module in nn.ModuleList which is SingleHeadAttention
            head_outputs.append(head_output)  # append output for the next step
        
        # 2. Concatenate outputs of heads
        # Concatenate the outputs for each heads for get single output
        # The outputs of one single head is equal to outputs of SingleHeadAttention
        concat_out = torch.cat(head_outputs, dim=-1)  # concatenate the list along dimension -1

        # 3. Feed it to the proj layer 
        out = self.proj(concat_out)  # This make linear projection to match the dimensionality

        ## ref: ChatGPT
        ## query:
        # TODO: Implement the multi-head attention forward pass.             #
        # Pass inputs to all heads, concatenate outputs of heads, and then   #
        # feed it to the projection layer to get the output tensor of the    #
        # same shape as the input.                                           #
        ######################################################################
        #                          END OF YOUR CODE                          #
        ######################################################################
        return out


class LayerNorm(nn.Module):
    def __init__(self, emb_dim: int, eps: float = 1e-10):
        """
        The class implements the layer normalization layer.
        Unlike batch normalization, it estimates the statistics along with
        the embedding dimension, hence independent to the batch dim statistics.
        For an input of shape (batch_size, seq_len, emb_dim), it computes
        the mean and standard deviation of shape (batch_size, seq_len) and
        use them for normalization.

        Args:
        - emb_dim: An integer giving the embedding dimension
        - eps: A floating point giving a small positive number
        """
        super().__init__()

        self.eps = eps
        self.gamma = None  # learnable scale
        self.beta = None  # learnable shift
        ######################################################################
        # TODO: Initialize the scale and shift parameters.                   #
        # Initialize the scale parameters to all ones and shift parameters   #
        # to all zeros. Note that parameters should be encapsulated by       #
        # nn.Parameter, such that they are iterated by parameters()          #
        ######################################################################
        # Replace "pass" with your code (do not modify this line)
        self.gamma = nn.Parameter(torch.ones(emb_dim), requires_grad=True)  
        self.beta = nn.Parameter(torch.zeros(emb_dim), requires_grad=True)
        # required_grad is necessary to make parameter learnable

        ## ref: No reference (my own code)
        ######################################################################
        #                          END OF YOUR CODE                          #
        ######################################################################

    def forward(self, x: Tensor):
        """
        The forward pass for the layer normalization layer.

        Args:
        - x: A Tensor of shape (batch_size, seq_len, emb_dim)

        Returns:
        - out: A Tensor of shape (batch_size, seq_len, emb_dim)
        """
        out = None
        ######################################################################
        # TODO: Implement the forward pass for the LayerNorm module.         #
        # Compute the mean and standard deviation of input and use these to  #
        # normalize the input. Then, use self.gamma and self.beta to scale   #
        # these and shift the normalized input.                              #
        # Do NOT use torch.std when computing the standard deviation.        #
        ######################################################################
        # Replace "pass" with your code (do not modify this line)
        ## 1. Compute the mean and standard deviation of input
        # mean
        mean = torch.mean(x, dim=-1, keepdim=True)  # dim=-1; mean along with embedding dimension
                                                    # keepdim=True; for broadcasting
        # std
        var = torch.mean((x - mean)**2, dim=-1, keepdim=True)  # MSE
        std = (var + self.eps) ** (0.5)

        ## 2. Normalize the input
        x_norm = (x - mean) / std

        ## 3. Scale and shift the normalize input with learnable parameters
        out = x_norm * self.gamma + self.beta

        ## ref: No reference (My own code)
        ######################################################################
        #                          END OF YOUR CODE                          #
        ######################################################################
        return out


class FFN(nn.Module):
    def __init__(self, inp_dim: int, hid_dim: int):
        super().__init__()
        """
        An implementation of a feed-forward network.
        The network has the following architecture:

        linear - relu - linear

        The input has the shape of (batch_size, seq_len, inp_dim)

        Args:
        - inp_dim: An integer giving the dimension of the input
        - hid_dim: An integer giving the dimension of the hidden layer
        """
        self.linear1 = None
        self.linear2 = None
        self.act = None
        ######################################################################
        # TODO: Initialize the two-layer feed-forward network. Figure out    #
        # an appropriate dimension of the output and apply it for the second #
        # fully-connected layer.                                             #
        ######################################################################
        # Replace "pass" with your code (do not modify this line)
        self.linear1 = nn.Linear(inp_dim, hid_dim)
        self.linear2 = nn.Linear(hid_dim, inp_dim)
        self.act = nn.ReLU()

        ## ref: No reference
        ######################################################################
        #                          END OF YOUR CODE                          #
        ######################################################################

    def forward(self, x):
        """
        The forward pass for FFN.

        Args:
        - x: A Tensor of shape (batch_size, seq_len, emb_dim)

        Returns:
        - out: A Tensor of shape (batch_size, seq_len, emb_dim)
        """
        out = None
        ######################################################################
        # TODO: Implement the forward pass for the FFN module.               #
        ######################################################################
        # Replace "pass" with your code (do not modify this line)
        out = self.act(self.linear1(x))  # first layer
        out = self.linear2(out)          # second layer

        ## ref: No reference
        ######################################################################
        #                          END OF YOUR CODE                          #
        ######################################################################
        return out


class EncoderBlock(nn.Module):
    def __init__(
        self, num_heads: int, emb_dim: int, ffn_dim: int, dropout: float = 0.
    ):
        """
        A naive implementation of a Transformer encoder block.
        The network has the following architecture:

        input - MultiHeadAttention - dropout (+ input) - norm1 - FFN
        - dropout (+ input) - norm2 - output

        The input has the shape of (batch_size, seq_len, emb_dim), and FFN gets
        ffn_dim as its hidden dimension.

        Args:
        - num_heads: An integer giving the number of heads
        - emb_dim: An integer giving the embedding dimension
        - ffn_dim: An integer giving the dimension of the hidden layer in FFN
        - dropout: A floating point giving the drop rate; 0 if no dropout
        """
        super().__init__()

        self.attn = None
        self.norm1 = None
        self.ffn = None
        self.norm2 = None
        self.dropout = None
        ######################################################################
        # TODO: Initialize the Transformer encoder block with followings:    #
        # 1. self.attn: multi-head attention. Be careful on the output dim   #
        # 2. self.norm1: layer normalization after multi-head attention      #
        # 3. self.ffn: feed-forward network with the hidden dimension of     #
        # 4. self.norm2: layer normalization after feed-forward network      #
        #    ffn_dim                                                         #
        # 5. self.dropout: dropout layer with the given dropout parameter    #
        ######################################################################
        # Replace "pass" with your code (do not modify this line)
        self.attn = MultiHeadAttention(num_heads, dim_in=emb_dim, dim_out=emb_dim)  # self-attention
        self.norm1 = LayerNorm(emb_dim)  # The dimension is maintained.
        self.ffn = FFN(inp_dim=emb_dim, hid_dim=ffn_dim)  # Output dim is equal to input dim
        self.norm2 = LayerNorm(emb_dim)  # The dimension is maintained.
        self.dropout = nn.Dropout(dropout)

        ## ref: No reference
        ######################################################################
        #                          END OF YOUR CODE                          #
        ######################################################################

    def forward(self, x):
        """
        The forward pass for a Transformer encoder block.

        Args:
        - x: A Tensor of shape (batch_size, seq_len, emb_dim)

        Returns:
        - out: A Tensor of shape (batch_size, seq_len, emb_dim)
        """
        out = None
        ######################################################################
        # TODO: Implement the forward pass for a Transformer encoder block.  #
        # For self-attention, the inputs (queries, keys, values) to the      #
        # attention module are the same.                                     #
        ######################################################################
        # Replace "pass" with your code (do not modify this line)
        """
        input - MultiHeadAttention - dropout (+ input) - norm1 - FFN
        - dropout (+ input) - norm2 - output
        """
        residual = x
        out1 = self.attn(x, x, x)             # MultiHeadAttention; self-attention
        out1 = self.dropout(out1) + residual  # dropout (+ input)
        out1 = self.norm1(out1)               # norm1

        residual = out1
        out2 = self.ffn(out1)                 # FFN
        out2 = self.dropout(out2) + residual  # dropout (+ input)
        out = self.norm2(out2)               # norm2

        ## ref: No reference
        ######################################################################
        #                          END OF YOUR CODE                          #
        ######################################################################
        return out


def get_subsequent_mask(seq):
    """
    An implementation of the decoder self-attention mask. This is used to
    mask out the target sequence to prohibit the decoder to look ahead the
    future input values when training the model.

    Args:
    - seq: A Tensor of shape (batch_size, seq_len) giving the input sequence

    Returns:
    - mask: A Tensor of shape (batch_size, seq_len, seq_len) giving the mask
    """
    mask = None
    ##########################################################################
    # TODO: Return mask that prohibit the decoder to look ahead the future   #
    # input values. The mask for each sequence in the batch should be a      #
    # boolean matrix, which has True for the place where we have to apply    #
    # mask and False where we don't have to apply the mask.                  #
    # Hint: torch.triu                                                       #
    ##########################################################################
    # Replace "pass" with your code (do not modify this line)
    N, K = seq.shape
    mask = torch.triu(torch.ones_like(seq).bool())[1:]  # make upper triangle matrix and remove the first row
    mask = mask.repeat(N, 1).reshape(N, K, K)  # repeat; repeat tensor `mask` N times
                                               # (N, K, K); we (K, K) mask for every batch size, N

    ## ref: No reference
    ##########################################################################
    #                            END OF YOUR CODE                            #
    ##########################################################################
    return mask


class DecoderBlock(nn.Module):
    def __init__(
        self, num_heads: int, emb_dim: int, ffn_dim: int, dropout: float = 0.
    ):
        """
        A naive implementation of a Transformer decoder block.
        The network has the following architecture:

        input - MaskedMultiHeadAttention - dropout (+ input) - norm1 - out1
        (enc_out, out1) - MultiHeadAttention - dropout (+ input) - norm2
        - FFN - dropout (+ input) - norm3 - output

        The first multi-head attention is masked self-attention, and
        the second multi-head attention is cross-attention,
        where query is the masked self-attention result of the decoder input,
        and keys and values are from the encoder output.

        The input has the shape of (batch_size, seq_len, emb_dim), and FFN gets
        ffn_dim as its hidden dimension.

        Args:
        - num_heads: An integer giving the number of heads
        - emb_dim: An integer giving the embedding dimension
        - ffn_dim: An integer giving the dimension of the hidden layer in FFN
        - dropout: A floating point giving the drop rate; 0 if no dropout
        """
        super().__init__()

        self.attn_mask = None
        self.norm1 = None
        self.attn_cross = None
        self.norm2 = None
        self.ffn = None
        self.norm3 = None
        self.dropout = None
        ######################################################################
        # TODO: Initialize the Transformer encoder block with followings:    #
        # 1. self.attn_mask: masked self-attention                           #
        # 2. self.norm1: layer normalization after masked self-attention     #
        # 3. self.attn_cross: cross-attention                                #
        # 4. self.norm2: layer normalization after cross-attention           #
        # 5. self.ffn: feed-forward network with the hidden dimension of     #
        # 6. self.norm3: layer normalization after feed-forward network      #
        # 7. self.dropout: dropout layer with the given dropout parameter    #
        ######################################################################
        # Replace "pass" with your code (do not modify this line)
        # 1. self.attn_mask: masked self-attention 
        self.attn_mask = MultiHeadAttention(num_heads, dim_in=emb_dim, dim_out=emb_dim)  # self-attn: in=out
        
        # 2. self.norm1: layer normalization after masked self-attention
        self.norm1 = LayerNorm(emb_dim)  # The dimension is maintained.
        
        # 3. self.attn_cross: cross-attention
        self.attn_cross = MultiHeadAttention(num_heads, dim_in=emb_dim, dim_out=emb_dim)
        
        # 4. self.norm2: layer normalization after cross-attention
        self.norm2 = LayerNorm(emb_dim)

        # 5. self.ffn: feed-forward network with the hidden dimension
        self.ffn = FFN(inp_dim=emb_dim, hid_dim=ffn_dim)

        # 6. self.norm3: layer normalization after feed-forward network
        self.norm3 = LayerNorm(emb_dim)

        # 7. self.dropout: dropout layer with the given dropout parameter 
        self.dropout = nn.Dropout(dropout)

        ## ref: No reference
        ######################################################################
        #                          END OF YOUR CODE                          #
        ######################################################################

    def forward(
        self, x: Tensor, enc_out: Tensor, mask: Tensor = None
    ) -> Tensor:
        """
        The forward pass for a Transformer decoder block.

        Args:
        - x: A Tensor of shape (batch_size, seq_len, emb_dim) giving
          the decoder input
        - enc_out: A Tensor of shape (batch_size, seq_len, emb_dim) giving
          the keys and values for the cross-attention from the encoder
        - mask: A boolean Tensor of shape (batch_size, seq_len, seq_len) giving
          the mask applied to the affinity matrix for masked attention

        Returns:
        - out: A Tensor of shape (batch_size, seq_len, emb_dim)
        """
        out = None
        ######################################################################
        # TODO: Implement the forward pass for a Transformer decoder block.  #
        # For the masked self-attention, the inputs (queries, keys, values)  #
        # to the attention module are the same. For the cross-attention, the #
        # queries are the output of the masked self-attention followed by    #
        # Add & Norm, and the keys and values are the output of the encoder. #
        ######################################################################
        # Replace "pass" with your code (do not modify this line)
        """
        input - MaskedMultiHeadAttention - dropout (+ input) - norm1 - out1
        (enc_out, out1) - MultiHeadAttention - dropout (+ input) - norm2
        - FFN - dropout (+ input) - norm3 - output

        The first multi-head attention is masked self-attention, and
        the second multi-head attention is cross-attention,
        where query is the masked self-attention result of the decoder input,
        and keys and values are from the encoder output.
        """
        residual = x
        out1 = self.attn_mask(x, x, x, mask)  # self-attn
        out1 = self.dropout(out1) + residual  # dropout (+ input)
        out1 = self.norm1(out1)               # norm1

        residual = out1
        out2 = self.attn_cross(out1, enc_out, enc_out)  # cross-attn; key, value is output from encoder
        out2 = self.dropout(out2) + residual  # dropout (+ input)      
        out2 = self.norm2(out2)               # norm2

        residual = out2
        out3 = self.ffn(out2)                 # FFN
        out3 = self.dropout(out3) + residual
        out = self.norm3(out3)

        ## ref: No reference
        ######################################################################
        #                          END OF YOUR CODE                          #
        ######################################################################
        return out


class Encoder(nn.Module):
    def __init__(
        self,
        num_heads: int,
        emb_dim: int,
        ffn_dim: int,
        num_layers: int,
        dropout: float = 0.
    ):
        """
        The class implements the Encoder by stacking EncoderBlock layers.

        Args:
        - num_heads: An integer giving the number of heads
        - emb_dim: An integer giving the embedding dimension
        - ffn_dim: An integer giving the dimension of the hidden layer in FFN
        - num_layers: An integer giving the number of EncoderBlock layers
        - dropout: A floating point giving the drop rate; 0 if no dropout
        """
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderBlock(num_heads, emb_dim, ffn_dim, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, src_seq: Tensor):
        for _layer in self.layers:
            src_seq = _layer(src_seq)
        return src_seq


class Decoder(nn.Module):
    def __init__(
        self,
        num_heads: int,
        emb_dim: int,
        ffn_dim: int,
        num_layers: int,
        vocab_len: int,
        dropout: float = 0.
    ):
        super().__init__()
        """
        The class implements the Decoder by stacking DecoderBlock layers,
        followed by a fully-connected layer to project to the vector of length
        vocab_len.

        Args:
        - num_heads: An integer giving the number of heads
        - emb_dim: An integer giving the embedding dimension
        - ffn_dim: An integer giving the dimension of the hidden layer in FFN
        - num_layers: An integer giving the number of DecoderBlock layers
        - vocab_len: An integer giving the length of the vocabulary
        - dropout: A floating point giving the drop rate; 0 if no dropout
        """

        self.layers = nn.ModuleList([
            DecoderBlock(num_heads, emb_dim, ffn_dim, dropout)
            for _ in range(num_layers)
        ])
        self.proj_to_vocab = nn.Linear(emb_dim, vocab_len)
        a = (6. / (emb_dim + vocab_len)) ** 0.5
        nn.init.uniform_(self.proj_to_vocab.weight, -a, a)

    def forward(self, tgt_seq: Tensor, enc_out: Tensor, mask: Tensor):
        out = tgt_seq.clone()
        for _layer in self.layers:
            out = _layer(out, enc_out, mask)
        out = self.proj_to_vocab(out)
        return out


class Transformer(nn.Module):
    def __init__(
        self,
        num_heads: int,
        emb_dim: int,
        ffn_dim: int,
        num_enc_layers: int,
        num_dec_layers: int,
        vocab_len: int,
        dropout: float = 0.,
    ):
        """
        The class implements the encoder-decoder Transformer model.
        This model gets four inputs:
        1. The input sequence, which is the input to the encoder
        2. The input positional encodings
        3. The target sequence, which is the input to the decoder
        4. The target positional encodings

        The input sequence is passed to the embedding layer, added to the
        input positional encodings, and then passed to the encoder.
        The target sequence is passed to the embedding layer, added to the
        target positional encodings, and then passed to the decoder, together
        with the output of the encoder.

        Args:
        - num_heads: An integer giving the number of heads
        - emb_dim: An integer giving the embedding dimension
        - ffn_dim: An integer giving the dimension of the hidden layer in FFN
        - num_enc_layers: An integer giving the number of EncoderBlock layers
        - num_dec_layers: An integer giving the number of DecoderBlock layers
        - vocab_len: An integer giving the length of the vocabulary
        - dropout: A floating point giving the drop rate; 0 if no dropout
        """
        super().__init__()

        self.emb_layer = None
        ######################################################################
        # TODO: Initialize the embedding layer mapping vocab_len to emb_dim. #
        ######################################################################
        # Replace "pass" with your code (do not modify this line)
        self.emb_layer = nn.Embedding(vocab_len, emb_dim)

        ######################################################################
        #                          END OF YOUR CODE                          #
        ######################################################################
        self.encoder = Encoder(
            num_heads, emb_dim, ffn_dim, num_enc_layers, dropout
        )
        self.decoder = Decoder(
            num_heads, emb_dim, ffn_dim, num_dec_layers, vocab_len, dropout
        )

    def forward(
        self, inp_seq: Tensor, inp_pos: Tensor, tgt_seq: Tensor,
        tgt_pos: Tensor
    ) -> Tensor:
        """
        The forward pass for a Transformer.

        Args:
        - inp_seq: A Tensor of shape (batch_size, seq_len) giving
          the input sequence,  which is the input to the encoder
        - inp_pos: A Tensor of shape (batch_size, seq_len, emb_dim) giving
          the positional encodings for the input sequence
        - tgt_seq: A Tensor of shape (batch_size, tgt_len) giving
          the target sequence, which is the input to the decoder
        - tgt_pos: A Tensor of shape (batch_size, tgt_len, emb_dim) giving
          the positional encodings for the target sequence

        Returns:
            out: A Tensor of shape (batch_size*tgt_len, emb_dim)
        """
        out = None
        ######################################################################
        # TODO: Implement the forward pass for a Transformer.                #
        # Specifically, you can follow the steps below:                      #
        # 1. Embed inputs and targets.                                       #
        # 2. Add positional encodings to the embedded inputs and targets.    #
        # 3. Encode the input embeddings.                                    #
        # 4. Generate a mask to prohibits the decoder to look ahead the      #
        # future along the target sequence dimension.                        #
        # 5. Decode the target embeddings with the encoder output and mask.  #
        # 6. Reshape the output of the decoder.                              #
        # Note that you need to eliminate the last sequence dimension of the #
        # target sequence, as there is nothing to predict from it.           #
        # Hint: get_subsequent_mask                                          #
        ######################################################################
        # Replace "pass" with your code (do not modify this line)
        # 1. Embed inputs and targets
        inp_embedding = self.emb_layer(inp_seq)
        tgt_embedding = self.emb_layer(tgt_seq[:, :-1])  # eleminate the last sequcence dimension

        # 2. Add positional encodings to the embedded inputs and targets
        inp_embedding = inp_embedding + inp_pos
        tgt_embedding = tgt_embedding + tgt_pos[:, :-1]  # eleminate the last sequcence dimension   

        # 3. Encode the input embeddings
        enc_out = self.encoder(inp_embedding)

        # 4. Generate a mask to prohibits the decoder to look ahead the future along the target sequence dimension
        mask = get_subsequent_mask(tgt_embedding)  # the mask should look tgt emb dimension

        # 5. Decode the target embeddings with the encoder output and mask
        dec_out = self.decoder(tgt_embedding, enc_out, mask)

        # 6. Reshape the output of the decoder
        out = dec_out.reshape(-1, dec_out.shape[-1])

        ## ref: ChatGPT
        # query:
        # I want to make transformer which follows
        # 1. Embed inputs and targets.                                       #
        # 2. Add positional encodings to the embedded inputs and targets.    #
        # 3. Encode the input embeddings.                                    #
        # 4. Generate a mask to prohibits the decoder to look ahead the      #
        # future along the target sequence dimension.                        #
        # 5. Decode the target embeddings with the encoder output and mask.  #
        # 6. Reshape the output of the decoder.                              #
        # Note that you need to eliminate the last sequence dimension of the #
        # target sequence, as there is nothing to predict from it.           #
        # Hint: get_subsequent_mask                                          #
        ######################################################################
        #                          END OF YOUR CODE                          #
        ######################################################################
        return out


def find_transformer_parameters():
    num_heads = 2
    emb_dim = 32
    ffn_dim = 32
    num_enc_layers = 2
    num_dec_layers = 2
    dropout = 0.3
    loss_func = CrossEntropyLoss
    pos_enc = pos_enc_simple
    num_epochs = 10
    warmup_interval = None
    warmup_lr = 6e-6
    lr = 1e-4
    ##########################################################################
    # TODO: Tune all parameters above so your model achieves 80% validation  #
    # accuracy.                                                              #
    ##########################################################################
    # Replace "pass" with your code (do not modify this line)
    pass
    ##########################################################################
    #                            END OF YOUR CODE                            #
    ##########################################################################
    return (
        num_heads,
        emb_dim,
        ffn_dim,
        num_enc_layers,
        num_dec_layers,
        dropout,
        loss_func,
        pos_enc,
        num_epochs,
        warmup_interval,
        warmup_lr,
        lr,
    )


class AddSubDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        input_seqs,
        target_seqs,
        token_dict,
        special_tokens,
        emb_dim,
        pos_enc,
    ):
        """
        The class implements the data loader used for the toy dataset.

        Args:
        - input_seqs: A list of input strings
        - target_seqs: A list of output strings
        - token_dict: A dictionary to convert input strings to tokens
        - special_tokens: A list of strings giving
        - emb_dim: Embedding dimension of the Transformer
        - pos_enc: A function to compute positional encodings
        """
        self.input_seqs = input_seqs
        self.target_seqs = target_seqs
        self.token_dict = token_dict
        self.emb_dim = emb_dim
        self.special_tokens = special_tokens
        self.pos_enc = pos_enc

    def preprocess(self, inp):
        return str2token(inp, self.token_dict, self.special_tokens)

    def __getitem__(self, idx):
        """
        The core function to get an element with index of idx in the data.

        Args:
        - idx: index of the element that we need to extract from the data

        Returns:
        - inp_prep: A Tensor of shape (seq_len,)
        - inp_pos_enc: A tensor of shape (seq_len, emb_dim)
        - tgt_prep: A Tensor of shape (out_seq_len,)
        - tgt_pos_enc: A tensor of shape (out_seq_len, emb_dim)
        """
        inp = self.input_seqs[idx]
        tgt = self.target_seqs[idx]
        inp_prep = torch.tensor(self.preprocess(inp))
        tgt_prep = torch.tensor(self.preprocess(tgt))
        inp_pos = len(inp_prep)
        inp_pos_enc = self.pos_enc(inp_pos, self.emb_dim)
        tgt_pos = len(tgt_prep)
        tgt_pos_enc = self.pos_enc(tgt_pos, self.emb_dim)

        return inp_prep, inp_pos_enc[0], tgt_prep, tgt_pos_enc[0]

    def __len__(self):
        return len(self.input_seqs)


def LabelSmoothingLoss(pred, gt):
    """
    Args:
    - pred: predicted tensor of shape (batch_size*tgt_len, vocab_len)
    - gt: ground truth tensor of shape (batch_size, tgt_len)
    """
    gt = gt.contiguous().view(-1)
    eps = 0.1
    n_class = pred.size(1)
    one_hot = F.one_hot(gt).to(pred.dtype)
    one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
    log_prb = F.log_softmax(pred, dim=1)
    loss = -(one_hot * log_prb).sum(dim=1)
    loss = loss.sum()
    return loss


def CrossEntropyLoss(pred, gt):
    """
    Args:
    - pred: predicted tensor of shape (batch_size*tgt_len, vocab_len)
    - gt: ground truth tensor of shape (batch_size, tgt_len)
    """
    loss = F.cross_entropy(pred, gt, reduction='sum')
    return loss
