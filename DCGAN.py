import torch
import torch.nn as nn
import torch.nn.functional as F
from SpectralNorm import *
torch.manual_seed(0)

## Random Normal Initialization
def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()

# G(z)
class DCGenerator(nn.Module):
    # initializers
    def __init__(self, d=128):
        super(DCGenerator, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(100, d*8, 4, 1, 0)
        self.deconv1_bn = nn.BatchNorm2d(d*8)
        self.deconv2 = nn.ConvTranspose2d(d*8, d*4, 4, 2, 1)
        self.deconv2_bn = nn.BatchNorm2d(d*4)
        self.deconv3 = nn.ConvTranspose2d(d*4, d*2, 4, 2, 1)
        self.deconv3_bn = nn.BatchNorm2d(d*2)
        self.deconv4 = nn.ConvTranspose2d(d*2, d, 4, 2, 1)
        self.deconv4_bn = nn.BatchNorm2d(d)
        self.deconv5 = nn.ConvTranspose2d(d, 1, 4, 2, 1)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input):
        # x = F.relu(self.deconv1(input))
        x = F.relu(self.deconv1_bn(self.deconv1(input)))
        x = F.relu(self.deconv2_bn(self.deconv2(x)))
        x = F.relu(self.deconv3_bn(self.deconv3(x)))
        x = F.relu(self.deconv4_bn(self.deconv4(x)))
        x = F.tanh(self.deconv5(x))

        return x

# D(z)
class DCDiscriminator(nn.Module):
    # initializers
    def __init__(self, d=128, spectral_norm=False):
        super(DCDiscriminator, self).__init__()
        self.spectral_norm = spectral_norm
        if self.spectral_norm:
          print("Applying Spectral Norm")
          self.conv1 = SpectralNorm(nn.Conv2d(1, d, 4, 2, 1))
          self.conv2 = SpectralNorm(nn.Conv2d(d, d*2, 4, 2, 1))
          self.conv3 = SpectralNorm(nn.Conv2d(d*2, d*4, 4, 2, 1))
          self.conv4 = SpectralNorm(nn.Conv2d(d*4, d*8, 4, 2, 1))
          self.conv5 = SpectralNorm(nn.Conv2d(d*8, 1, 4, 1, 0))          
        else:
          print("Without Spectral Norm")
          self.conv1 = nn.Conv2d(1, d, 4, 2, 1)
          self.conv2 = nn.Conv2d(d, d*2, 4, 2, 1)
          self.conv2_bn = nn.BatchNorm2d(d*2)
          self.conv3 = nn.Conv2d(d*2, d*4, 4, 2, 1)
          self.conv3_bn = nn.BatchNorm2d(d*4)
          self.conv4 = nn.Conv2d(d*4, d*8, 4, 2, 1)
          self.conv4_bn = nn.BatchNorm2d(d*8)
          self.conv5 = nn.Conv2d(d*8, 1, 4, 1, 0)


    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input):
      if self.spectral_norm:
        x = F.leaky_relu(self.conv1(input), 0.2)
        x = F.leaky_relu(self.conv2(x), 0.2)
        x = F.leaky_relu(self.conv3(x), 0.2)
        x = F.leaky_relu(self.conv4(x), 0.2)
        x = F.sigmoid(self.conv5(x))
      else: # applying batch normalization
        x = F.leaky_relu(self.conv1(input), 0.2)
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.conv4_bn(self.conv4(x)), 0.2)
        x = F.sigmoid(self.conv5(x))

      return x

def get_disc_loss(G, D, criterion, x_, batch_size, device):
    '''
    Return the loss of the discriminator given inputs.
    Parameters:
        G: the generator model, which returns an image given z-dimensional noise
        D: the discriminator model, which returns a single-dimensional prediction of real/fake
        criterion: the loss function, which should be used to compare 
               the discriminator's predictions to the ground truth reality of the images 
               (e.g. fake = 0, real = 1)
        x_: a batch of real images
        batch_size: the number of images the generator should produce, 
                which is also the length of the real images
        device: the device type
    Returns:
        disc_loss: a torch scalar loss value for the current batch
    '''
    y_real_ = torch.ones(batch_size).to(device)
    y_fake_ = torch.zeros(batch_size).to(device)
    x_ = Variable(x_.to(device))

    D_result = D(x_).squeeze()
    D_real_loss = criterion(D_result, y_real_)

    z_ = torch.randn((batch_size, 100)).view(-1, 100, 1, 1)
    z_ = Variable(z_.to(device))
    G_result = G(z_)

    D_result = D(G_result).squeeze()
    D_fake_loss = criterion(D_result, y_fake_)
    D_fake_score = D_result.data.mean()

    D_train_loss = D_real_loss + D_fake_loss
    return D_train_loss

def get_gen_loss(G, D, criterion, batch_size, device):
    '''
    Return the loss of the generator given inputs.
    Parameters:
        gen: the generator model, which returns an image given z-dimensional noise
        disc: the discriminator model, which returns a single-dimensional prediction of real/fake
        criterion: the loss function, which should be used to compare 
               the discriminator's predictions to the ground truth reality of the images 
               (e.g. fake = 0, real = 1)
        num_images: the number of images the generator should produce, 
                which is also the length of the real images
        device: the device type
    Returns:
        gen_loss: a torch scalar loss value for the current batch
    '''
    y_real_ = torch.ones(batch_size).to(device)

    z_ = torch.randn((batch_size, 100)).view(-1, 100, 1, 1).to(device)

    G_result = G(z_)
    D_result = D(G_result).squeeze()
    G_train_loss = criterion(D_result, y_real_)
    return G_train_loss


