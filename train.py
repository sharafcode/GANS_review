import time
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from tqdm import tqdm as tq
from torchvision.utils import make_grid

## LinearGAN network
from GAN import *

## DCGAN network
from DCGAN import *


class Train():
  def __init__(self, is_linear, batch_norm, spectral_norm, z_dim, device='cuda'):
    self.is_linear = is_linear
    self.batch_norm = batch_norm
    self.spectral_norm = spectral_norm
    self.z_dim = z_dim
    self.device = device

    if self.is_linear:
      ## Linear Generator initialization
      self.G = LinearGenerator(self.z_dim, is_norm=self.batch_norm).to(self.device)
      ## Linear Discrimnator initialization
      self.D = LinearDiscriminator().to(self.device)
    
    else:
      ## Conv. Generator initialization
      self.G = DCGenerator(self.z_dim)
      self.G.weight_init(mean=0.0, std=0.02)
      self.G.to(self.device)

      ## Conv Discrimnator initialization
      self.D = DCDiscriminator(self.z_dim, spectral_norm=self.spectral_norm)
      self.D.weight_init(mean=0.0, std=0.02)
      self.D.to(self.device)

  def show_tensor_images(self, image_tensor, num_images=25, size=(1, 28, 28), is_linear = True):
    '''
    Function for visualizing images: Given a tensor of images, number of images, and
    size per image, plots and prints the images in a uniform grid.
    '''
    image_tensor = image_tensor.detach().cpu()
    if is_linear:
      image_tensor = image_tensor.view(-1, *size) ## Reshape the 1D images to 2D images again

    image_grid = make_grid(image_tensor[:num_images], nrow=5)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.show()

  def load_data(self, img_size, batch_size):
    # data_loader
    transform = transforms.Compose([
            transforms.Scale(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,), std=(0.5,))
    ])
    if self.is_linear:
      train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('.', download=True, transform=transforms.ToTensor()),
        batch_size=batch_size,
        shuffle=True)
    else:
      train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=True, download=True, transform=transform),
        batch_size=batch_size, shuffle=True)
    
    return train_loader

  def trainer(self, lr, batch_size = 128, n_epochs= 20, img_size = 64, display_step=100, ):
    G = self.G
    D = self.D

    criterion = nn.BCEWithLogitsLoss()
    train_loader = self.load_data(img_size, batch_size)
    
    # set_trace()
    
    ## Adam optimizers
    G_optimizer = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    
    if self.spectral_norm:  # Spectral Normalization optimization
      D_optimizer = optim.Adam(filter(lambda p: p.requires_grad, D.parameters()), lr=lr, betas=(0.0,0.9))
    else:
      D_optimizer = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
    
    mean_generator_loss = 0
    mean_discriminator_loss = 0
    num_iter = 0

    print('Training start!')
    for epoch in tq(range(n_epochs)):
        epoch_start_time = time.time()
        for x_, _ in train_loader:
            batch_size = len(x_) #x_.size()[0]

            if self.is_linear: # Flatten the batch of real images from the dataset  
              x_ = x_.view(batch_size, -1).to(self.device)

            ### Train discriminator D
            D.zero_grad()

            if self.is_linear:
              D_train_loss = linear_disc_loss(G, D, criterion, x_, batch_size, self.z_dim, self.device)
            else:
              D_train_loss = get_disc_loss(G, D, criterion, x_, batch_size, self.device)
            
            # Update Gradients
            D_train_loss.backward(retain_graph=True)
            D_optimizer.step()

            ### Train generator G
            G.zero_grad()

            if self.is_linear:
              G_train_loss = linear_gen_loss(G, D, criterion, batch_size, self.z_dim, self.device)
            else:
              G_train_loss = get_gen_loss(G, D, criterion, batch_size, self.device)
            
            # Update Gradients
            G_train_loss.backward()
            G_optimizer.step()

            # Keep track of the average discriminator loss
            mean_discriminator_loss += D_train_loss.item() / display_step

            # Keep track of the average generator loss
            mean_generator_loss += G_train_loss.item() / display_step

            ### Visualization code for steps ###
            if num_iter % display_step == 0 and num_iter > 0:
                print(f"Step {num_iter}: Generator loss: {mean_generator_loss}, discriminator loss: {mean_discriminator_loss}")
                if self.is_linear:
                  fake_noise = get_noise(batch_size, self.z_dim, device= self.device)
                else:
                  fake_noise = torch.randn((batch_size, 100)).view(-1, 100, 1, 1).to(self.device)
                fake = G(fake_noise)

                self.show_tensor_images(fake, is_linear = self.is_linear)
                self.show_tensor_images(x_, is_linear = self.is_linear)
                mean_generator_loss = 0
                mean_discriminator_loss = 0
            num_iter += 1

