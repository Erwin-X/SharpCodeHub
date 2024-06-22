import torch
from torch import nn
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms as transforms
import numpy as np
import torch.optim as optim
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(
          self, 
          x_dim,
          hidden_dim,
          z_dim
        ):
        super(VAE, self).__init__()

        # Define autoencoding layers
        self.enc_layer1 = nn.Linear(x_dim, hidden_dim)
        self.enc_layer2_mu = nn.Linear(hidden_dim, z_dim)
        self.enc_layer2_logvar = nn.Linear(hidden_dim, z_dim)

        # Define autoencoding layers
        self.dec_layer1 = nn.Linear(z_dim, hidden_dim)
        self.dec_layer2 = nn.Linear(hidden_dim, x_dim) 

    def encoder(self, x):
        x = F.relu(self.enc_layer1(x))
        mu = F.relu(self.enc_layer2_mu(x))
        logvar = F.relu(self.enc_layer2_logvar(x))
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(logvar/2)
        eps = torch.randn_like(std)
        z = mu + std * eps
        return z

    def decoder(self, z):
        # Define decoder network
        output = F.relu(self.dec_layer1(z))
        output = F.relu(self.dec_layer2(output))
        return output

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        output = self.decoder(z)
        return output, z, mu, logvar

# Define the loss function
def loss_function(output, x, mu, logvar):
    batch_size = x.shape[0]
    recon_loss = F.mse_loss(output, x, reduction='sum') / batch_size
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + 0.002  * kl_loss


def train_model(
    train_loader=None,
    learning_rate=1e-4, 
    num_epochs=15,
    input_dim=28*28,
    hidden_dim=256,
    latent_dim=64
  ):
  # Define the VAE model
  model = VAE(x_dim=input_dim, hidden_dim=hidden_dim, z_dim=latent_dim)

  # Define the optimizer
  optimizer = optim.Adam(model.parameters(), lr=learning_rate)
  
  # Train the model
  for epoch in range(num_epochs):
      epoch_loss = 0
      for batch_idx, batch in enumerate(train_loader):
          # Zero the gradients
          optimizer.zero_grad()

          # Get batch
          x = batch[0]
          x = x.reshape(x.shape[0], -1)

          # Forward pass
          output, z, mu, logvar = model(x)

          # Calculate loss
          loss = loss_function(output, x, mu, logvar)

          # Backward pass
          loss.backward()

          # Update parameters
          optimizer.step()

          # Add batch loss to epoch loss
          epoch_loss += loss.item()

          if batch_idx % 20 == 0:
              print(f"Epoch: {epoch+1}/{num_epochs}, Batch: {batch_idx}, Loss: {loss}")

      # Print epoch loss
      print(f"Epoch: {epoch+1}/{num_epochs}, Loss: {epoch_loss/(batch_idx*len(batch))}")

  return model
