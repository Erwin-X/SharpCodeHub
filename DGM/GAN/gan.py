import torch
from torch import nn
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms as transforms
import numpy as np
import torch.optim as optim
import torch.nn.functional as F

class GAN(nn.Module):
    def __init__(self, g_input_dim=64, d_input_dim=28*28,  hidden_dim=256):
        super(GAN, self).__init__()
        self.generator = nn.Sequential(
            nn.Linear(g_input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim*2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim*2, d_input_dim),
            nn.Tanh()
        )

        self.discriminator = nn.Sequential(
            nn.Linear(d_input_dim, hidden_dim*2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def generator_forward(self, z):
        return self.generator(z)

    def discriminator_forward(self, x):
        return self.discriminator(x)

    def generate(self, z):
        return self.generator_forward(z)


def train_model(
    train_loader=None,
    learning_rate=2e-4, 
    num_epochs=15,
    g_input_dim=64,
    d_input_dim=28*28,
    hidden_dim=256
  ):
  # Loss
  criterion = nn.BCELoss()

  # Define the GAN model
  model = GAN(g_input_dim=g_input_dim, d_input_dim=d_input_dim, hidden_dim=hidden_dim)

  # Define the optimizer
  g_optimizer = optim.Adam(model.generator.parameters(), lr=learning_rate)
  d_optimizer = optim.Adam(model.discriminator.parameters(), lr=learning_rate)

  def d_train(x):
    #=======================Train the discriminator=======================#
    d_optimizer.zero_grad()

    # train discriminator on real
    x_real, y_real = x, torch.ones(x.shape[0], 1)
    d_output = model.discriminator_forward(x_real)
    d_real_loss = criterion(d_output, y_real)
    d_real_score = d_output

    # train discriminator on fack
    z = torch.randn(x.shape[0], g_input_dim)
    x_fake, y_fake = model.generator_forward(z), torch.zeros(x.shape[0], 1)

    d_fake_output = model.discriminator_forward(x_fake)
    d_fake_loss = criterion(d_fake_output, y_fake)
    d_fake_score = d_fake_output

    # gradient backprop & optimize ONLY D's parameters
    d_loss = d_real_loss + d_fake_loss
    d_loss.backward()
    d_optimizer.step()

    return  d_loss.item()

  def g_train(x):
    #=======================Train the generator=======================#
    g_optimizer.zero_grad()

    z = torch.randn(x.shape[0], g_input_dim)
    y = torch.ones(x.shape[0], 1)

    g_output = model.generator_forward(z)
    d_output = model.discriminator_forward(g_output)
    g_loss = criterion(d_output, y)

    # gradient backprop & optimize ONLY G's parameters
    g_loss.backward()
    g_optimizer.step()

    return g_loss.item()


  # Train GAN
  for epoch in range(num_epochs):   
      D_losses, G_losses = [], []
      for batch_idx, (x, _) in enumerate(train_loader):
          x = x.reshape(x.shape[0], -1)
          d_loss = d_train(x)
          g_loss = g_train(x)
          D_losses.append(d_loss)
          G_losses.append(g_loss)
        #   if batch_idx % 100 == 0:
        #       print(f"Epoch: {epoch+1}/{num_epochs}, Batch: {batch_idx}, D_loss: {d_loss}, G_loss: {g_loss}")

      print(f"Epoch: {epoch+1}/{num_epochs}, D_loss: {torch.mean(torch.FloatTensor(D_losses))}, G_loss: {torch.mean(torch.FloatTensor(G_losses))}")

  return model
