import torch
import torch.nn as nn
import torch.nn.functional as F

from typeguard import typechecked
from torchtyping import TensorType, patch_typeguard
from typing import Tuple

from .decoder import Decoder
from .encoder import Encoder

patch_typeguard()

class VAE(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, hidden_dim: int=100):
        """Initialize the VAE model.
        
        Args:
            obs_dim (int): Dimension of the observed data x, int
            latent_dim (int): Dimension of the latent variable z, int
            hidden_dim (int): Hidden dimension of the encoder/decoder networks, int
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = Encoder(input_dim, latent_dim, hidden_dim=hidden_dim)
        self.decoder = Decoder(input_dim, latent_dim, hidden_dim=hidden_dim)
    
    @typechecked
    def sample_with_reparametrization(self, mu: TensorType['batch_size', 'latent_dim'], 
                                      logsigma: TensorType['batch_size', 'latent_dim']) -> TensorType['batch_size', 'latent_dim']:
        """Draw sample from q(z) with reparametrization.
        
        We draw a single sample z_i for each data point x_i.
        
        Args:
            mu: Means of q(z) for the batch, shape [batch_size, latent_dim]
            logsigma: Log-sigmas of q(z) for the batch, shape [batch_size, latent_dim]
        
        Returns:
            z: Latent variables samples from q(z), shape [batch_size, latent_dim]
        """
        ##########################################################
        # YOUR CODE HERE
        sigma = torch.exp(logsigma)
        std_norm = torch.randn_like(sigma)
        return std_norm * sigma + mu
        ##########################################################
    
    @typechecked
    def kl_divergence(self, mu: TensorType['batch_size', 'latent_dim'], logsigma: TensorType['batch_size', 'latent_dim']) -> TensorType['batch_size']:
        """Compute KL divergence KL(q_i(z)||p(z)) for each q_i in the batch.
        
        Args:
            mu: Means of the q_i distributions, shape [batch_size, latent_dim]
            logsigma: Logarithm of standard deviations of the q_i distributions,
                      shape [batch_size, latent_dim]
        
        Returns:
            kl: KL divergence for each of the q_i distributions, shape [batch_size]
        """
        ##########################################################
        # YOUR CODE HERE
        log_sigma_squared = logsigma + logsigma
        sigma_squared = torch.exp(log_sigma_squared)
        mu_squared = mu * mu

        return 0.5 * torch.sum(sigma_squared + mu_squared - log_sigma_squared - 1, 1)
        ##########################################################
    
    @typechecked
    def elbo(self, x: TensorType['batch_size', 'input_dim']) -> TensorType['batch_size']:
        """Estimate the ELBO for the mini-batch of data.
        
        Args:
            x: Mini-batch of the observations, shape [batch_size, input_dim]
        
        Returns:
            elbo_mc: MC estimate of ELBO for each sample in the mini-batch, shape [batch_size]
        """
        ##########################################################
        # YOUR CODE HERE
        mu, logsigma = self.encoder(x)
        z = self.sample_with_reparametrization(mu, logsigma)
        theta = self.decoder(z)
        recon_loss = - F.binary_cross_entropy(theta, x, reduction="none").sum(-1)
        kl_loss = self.kl_divergence(mu, logsigma) 
        elbo_mc = recon_loss - kl_loss
        assert elbo_mc.shape == (x.shape[0],)
        return elbo_mc
        ##########################################################
        
    @typechecked
    def sample(self, num_samples: int, device: str='cpu') -> Tuple[
        TensorType['num_samples', 'latent_dim'],
        TensorType['num_samples', 'input_dim'],
        TensorType['num_samples', 'input_dim']]:
        """Generate new samples from the model.
        
        Args:
            num_samples: Number of samples to generate.
        
        Returns:
            z: Sampled latent codes, shape [num_samples, latent_dim]
            theta: Parameters of the output distribution, shape [num_samples, input_dim]
            x: Corresponding samples generated by the model, shape [num_samples, input_dim]
        """
        ##########################################################
        # YOUR CODE HERE
        z = torch.randn((num_samples, self.latent_dim), device=device)
        theta = self.decoder(z)
        x = torch.bernoulli(theta)
        return z, theta, x
        ##########################################################