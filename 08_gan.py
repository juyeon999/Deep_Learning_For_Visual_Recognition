"""
Implements GAN in PyTorch.
WARNING: you SHOULD NOT use ".to()" or ".cuda()" in each implementation block.
"""
import torch
from torch import nn
from torch.nn import functional as F


def hello():
    """
    This is a sample function that we will try to import and run to ensure
    that our environment is correctly set up on Google Colab.
    """
    print("Hello from gan.py!")


def sample_noise(batch_size, noise_size):
    """
    Generate a PyTorch Tensor of uniform random noise in the range [-1,1].

    Input:
    - batch_size: Python integer giving the batch size of noise to generate.
    - noise_size: Python integer giving the dimension of noise to generate.

    Output:
    - Tensor of shape (batch_size, noise_size) containing uniform random
      noise in the range [-1,1].
    """
    noise = None
    ##########################################################################
    # TODO: Implement sample_noise using torch.rand                          #
    ##########################################################################
    # Replace "pass" with your code (do not modify this line)
    noise = 2 * torch.rand((batch_size, noise_size)) - 1
    ##########################################################################
    #                            END OF YOUR CODE                            #
    ##########################################################################
    return noise


def get_discriminator(input_size, hidden_dim):
    """
    Build and return a PyTorch nn.Sequential model for the discriminator.
    """

    model = None
    ##########################################################################
    # TODO: Implement the fully-connected discriminator architecture.        #
    # The network gets a batch of input images of shape (N ,D), and          #
    # the following layers map it to hidden features of shape (N, H), and    #
    # the final layer maps to it to a scalar value (N, 1).                   #
    # Wrap all layers with nn.Sequential and assign it to model.             #
    # Hint: nn.Linear, nn.LeakyReLU                                          #
    ##########################################################################
    # Replace "pass" with your code (do not modify this line)
    model = nn.Sequential(
        nn.Linear(input_size, hidden_dim),
        nn.LeakyReLU(0.01),
        nn.Linear(hidden_dim, hidden_dim),
        nn.LeakyReLU(0.01),
        nn.Linear(hidden_dim, 1)
        )
    ##########################################################################
    #                            END OF YOUR CODE                            #
    ##########################################################################
    return model


def get_generator(noise_size, hidden_dim, input_size):
    """
    Build and return a PyTorch nn.Sequential model for the generator.
    """

    model = None
    ##########################################################################
    # TODO: Implement the fully-connected generator architecture.            #
    # The network gets a batch of latent vectors of shape (N, Z) as input,   #
    # and outputs a tensor of estimated images of shape (N, D).              #
    # Wrap all layers with nn.Sequential and assign it to model.             #
    # Hint: nn.Linear, nn.ReLU, nn.Tanh                                      #
    ##########################################################################
    # Replace "pass" with your code (do not modify this line)
    model = nn.Sequential(
        nn.Linear(noise_size, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, input_size),
        nn.Tanh()
        )
    ##########################################################################
    #                            END OF YOUR CODE                            #
    ##########################################################################
    return model


def discriminator_loss(logits_real, logits_fake):
    """
    Computes the discriminator loss.

    Inputs:
    - logits_real: Tensor of shape (N,) giving scores for the real data.
    - logits_fake: Tensor of shape (N,) giving scores for the fake data.

    Returns:
    - loss: Tensor of scalar giving the loss for the discriminator.
    """
    loss = None
    ##########################################################################
    # TODO: Compute the discriminator loss.                                  #
    # Hint: F.binary_cross_entropy_with_logits,                              #
    # torch.zeros_like, torch.ones_like                                      #
    ##########################################################################
    # Replace "pass" with your code (do not modify this line)
    # For discriminator, the size and device of labels should be matched with real data
    true_labels = torch.ones(logits_real.shape, device=logits_real.device)
    fake_labels = torch.ones(logits_real.shape, device=logits_real.device)

    # binary_cross_entropy_with logits function gives cross entropy loss between logit(data) and labels.
    loss_true = F.binary_cross_entropy_with_logits(logits_real, true_labels)
    loss_fake = F.binary_cross_entropy_with_logits(logits_fake, fake_labels)

    loss = loss_true + loss_fake
    ##########################################################################
    #                            END OF YOUR CODE                            #
    ##########################################################################
    return loss


def generator_loss(logits_fake):
    """
    Computes the generator loss.

    Inputs:
    - logits_fake: Tensor of shape (N,) giving scores for the fake data.

    Returns:
    - loss: Tensor of scalar giving the loss for the generator.
    """
    loss = None
    ##########################################################################
    # TODO: Compute the generator loss.                                      #
    # Hint: F.binary_cross_entropy_with_logits, torch.ones_like              #
    ##########################################################################
    # Replace "pass" with your code (do not modify this line)
    # For generator, the size and device of labels should be matched with fake data
    fake_labels = torch.ones(size=logits_fake.shape, device=logits_fake.device)
    loss = F.binary_cross_entropy_with_logits(logits_fake, fake_labels)
    ##########################################################################
    #                            END OF YOUR CODE                            #
    ##########################################################################
    return loss


def ls_discriminator_loss(logits_real, logits_fake):
    """
    Compute the discriminator loss for Least Squares GAN.

    Inputs:
    - logits_real: Tensor of shape (N,) giving scores for the real data.
    - logits_fake: Tensor of shape (N,) giving scores for the fake data.

    Returns:
    - loss: Tensor of scalar giving the loss for the discriminator.
    """
    loss = None
    ##########################################################################
    # TODO: Compute the discriminator loss for Least Squares GAN.            #
    ##########################################################################
    # Replace "pass" with your code (do not modify this line)
    # Real loss: 0.5 * E[(D(x)-1)**2]
    loss_real = 0.5 * torch.mean((logits_real - 1) ** 2)
    # Fake loss: 0.5 * E[D(G(z))**2]
    loss_fake = 0.5 * torch.mean(logits_fake ** 2)

    loss = loss_real + loss_fake
    ##########################################################################
    #                            END OF YOUR CODE                            #
    ##########################################################################
    return loss


def ls_generator_loss(logits_fake):
    """
    Compute the generator loss for Least Squares GAN.

    Inputs:
    - logits_fake: Tensor of shape (N,) giving scores for the fake data.

    Returns:
    - loss: Tensor of scalar giving the loss for the generator.
    """
    loss = None
    ##########################################################################
    # TODO: Compute the generator loss for Least Squares GAN.                #
    ##########################################################################
    # Replace "pass" with your code (do not modify this line)
    loss = 0.5 * torch.mean((logits_fake - 1) ** 2)
    ##########################################################################
    #                            END OF YOUR CODE                            #
    ##########################################################################
    return loss


def get_dc_discriminator():
    """
    Build and return a PyTorch nn.Sequential model for the DCGAN discriminator.
    """
    model = None
    ##########################################################################
    # TODO: Implement the deep convolutional GAN discriminator architecture. #
    # Hint: nn.Unflatten                                                     #
    ##########################################################################
    # Replace "pass" with your code (do not modify this line)
    alpha = 0.01
    model = nn.Sequential(
        nn.Unflatten(1, (1, 28, 28)),  # to make (N, 1, 28, 28) i.e. channel 1
        nn.Conv2d(1, 32, 5),           # (N, 32, 24, 24), 24 = 28 - 5 + 1
        nn.LeakyReLU(alpha),           
        nn.MaxPool2d(2, 2),            # (N, 32, 12, 12), H and W are halved
        nn.Conv2d(32, 64, 5),          # (N, 64, 8, 8), 8 = 12 - 5 + 1
        nn.LeakyReLU(alpha),           
        nn.MaxPool2d(2, 2),            # (N, 64, 4, 4), Halved
        nn.Flatten(),                  # (N, 64 * 4 * 4)
        nn.Linear(64 * 4 * 4, 4 * 4 * 64),
        nn.LeakyReLU(alpha),
        nn.Linear(4 * 4 * 64, 1)
    )
    ##########################################################################
    #                            END OF YOUR CODE                            #
    ##########################################################################
    return model


def get_dc_generator(noise_size):
    """
    Build and return a PyTorch nn.Sequential model for the DCGAN generator.
    """
    model = None
    ##########################################################################
    # TODO: Implement the deep convolutional GAN generator architecture.     #
    # Hint: nn.Unflatten, nn.ConvTranspose2d                                 #
    ##########################################################################
    # Replace "pass" with your code (do not modify this line)
    model = nn.Sequential(
        nn.Linear(noise_size, 1024),  
        nn.ReLU(),
        nn.BatchNorm1d(1024), 
        nn.Linear(1024, 7 * 7 * 128),
        nn.ReLU(),
        nn.BatchNorm1d(7 * 7 * 128),
        nn.Unflatten(1, (128, 7, 7)),  # C=128, H=W=7
        nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(64),
        nn.ConvTranspose2d(in_channels=64, out_channels=1, kernel_size=4, stride=2, padding=1),
        nn.Tanh(),
        nn.Flatten()
    )
    ##########################################################################
    #                            END OF YOUR CODE                            #
    ##########################################################################
    return model
