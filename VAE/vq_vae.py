import torch
from torch import nn
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms as transforms
import numpy as np
import torch.optim as optim
import torch.nn.functional as F

class VQVAE(nn.Module):
    def __init__(self,
            codebook_size=4,
            codebook_embedding_dim=2):
        super(VQVAE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 4, 4, stride=2, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(),
        )

        self.pre_quant_conv = nn.Conv2d(4, 2, kernel_size=1)
        self.embedding = nn.Embedding(num_embeddings=codebook_size, embedding_dim=codebook_embedding_dim)
        self.post_quant_conv = nn.Conv2d(2, 4, kernel_size=1)

        # Commitment Loss Beta
        self.beta = 0.2

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(4, 16, 4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 4, stride=2, padding=1),
            nn.Tanh(),
        )

        self.criterion = nn.MSELoss()

    def encode(self, x):
        # B, C, H, W
        encoded_output = self.encoder(x)
        quant_input = self.pre_quant_conv(encoded_output)
        return quant_input

    def vector_quantize(self, quant_input):
        ## Quantization
        B, C, H, W = quant_input.shape
        quant_input = quant_input.permute(0, 2, 3, 1)
        quant_input = quant_input.reshape((quant_input.size(0), -1, quant_input.size(-1)))

        # Compute pairwise distances
        dist = torch.cdist(quant_input, self.embedding.weight[None, :].repeat((quant_input.size(0), 1, 1)))

        # Find index of nearest embedding
        min_encoding_indices = torch.argmin(dist, dim=-1)  # shape: (B*H*W,)

        # Select the embedding weights
        quant_out = torch.index_select(self.embedding.weight, 0, min_encoding_indices.view(-1))  # shape: (B*H*W, codebook_embedding_dim)
        quant_input = quant_input.reshape((-1, quant_input.size(-1)))

        # Compute losses
        codebook_loss = torch.mean((quant_out - quant_input.detach())**2)
        commitment_loss = torch.mean((quant_out.detach() - quant_input)**2)
        quantize_losses = codebook_loss + self.beta*commitment_loss

        # Ensure straight through gradient: a trick for gradient backprop between quant_out -> quant_input
        quant_out = quant_input + (quant_out - quant_input).detach()
    
        # Reshaping back to original input shape
        quant_out = quant_out.reshape((B, H, W, C)).permute(0, 3, 1, 2)

        return quant_out, quantize_losses, codebook_loss, commitment_loss

    def decode(self, quant_out):
        ## Decoder part
        decoder_input = self.post_quant_conv(quant_out)
        output = self.decoder(decoder_input)
        return output

    def forward(self, x):
        quant_input = self.encode(x)
        quant_out, quantize_loss, codebook_loss, commitment_loss = self.vector_quantize(quant_input)
        output = self.decode(quant_out)
        recon_loss = self.criterion(output, x)
        total_loss = recon_loss + quantize_loss
        return output, total_loss, (recon_loss, codebook_loss, commitment_loss)


def train_model(
    train_loader=None,
    learning_rate=1e-4, 
    num_epochs=15,
    codebook_size=4,
    codebook_embedding_dim=2,
  ):
  # Define the VAE model
  model = VQVAE(codebook_size=codebook_size, codebook_embedding_dim=codebook_embedding_dim)

  # Define the optimizer
  optimizer = optim.Adam(model.parameters(), lr=learning_rate)
  
  # Train the model
  for epoch in range(num_epochs):
      epoch_loss = 0
      epoch_recon_loss = 0
      epoch_codebook_loss = 0
      epoch_commitment_loss = 0
      for batch_idx, batch in enumerate(train_loader):
          # Zero the gradients
          optimizer.zero_grad()

          # Get batch
          x = batch[0]

          # Forward pass
          output, loss, (recon_loss, codebook_loss, commitment_loss) = model(x)

          # Backward pass
          loss.backward()

          # Update parameters
          optimizer.step()

          # Add batch loss to epoch loss
          epoch_loss += loss.item()
          epoch_recon_loss += recon_loss.item()
          epoch_codebook_loss += codebook_loss.item()
          epoch_commitment_loss += commitment_loss.item()

          if batch_idx % 100 == 0:
              print(f"Epoch: {epoch+1}/{num_epochs}, Batch: {batch_idx}, Total loss: {loss:.2f}, Recon loss: {recon_loss:.2f}, Codebook loss: {codebook_loss:.2f}, Commitment loss: {commitment_loss:.2f}")

      # Print epoch loss
      print(f"Epoch: {epoch+1}/{num_epochs}, Total loss: {epoch_loss/(batch_idx*len(batch)):.2f}, Recon loss: {epoch_recon_loss/(batch_idx*len(batch)):.2f}, Codebook loss: {epoch_codebook_loss/(batch_idx*len(batch)):.2f}, Commitment loss: {epoch_commitment_loss/(batch_idx*len(batch)):.2f}")

  return model
