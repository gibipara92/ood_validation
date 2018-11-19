from __future__ import print_function
import numpy as np
import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F

def idx2onehot(labels, num_classes):
    """Embedding labels to one-hot form.

    Args:
      labels: (LongTensor) class labels, sized [N,].
      num_classes: (int) number of classes.

    Returns:
      (tensor) encoded labels, sized [N, #classes].
    """
    y = torch.eye(num_classes)
    return y[labels]


class VAE(nn.Module):
    def __init__(self, device, lat_dim=20):
        super(VAE, self).__init__()
        self.lat_dim = lat_dim
        self.device = device
        num_classes = 10
        #Encoder
        self.conv1e = nn.Conv2d(1,  64, kernel_size=4, stride=2)
        self.conv2e = nn.Conv2d(64,  64, kernel_size=4, stride=2)
        self.conv3e = nn.Conv2d(64,  64, kernel_size=4, stride=2)
        self.fc21 = nn.Linear(256, self.lat_dim)
        self.fc22 = nn.Linear(256, self.lat_dim)

        #Decoder
        self.fc3 = nn.Linear(self.lat_dim + num_classes, 64*4*4)
        self.conv1d = nn.ConvTranspose2d(64, 64, kernel_size=4, padding=1, stride=2)
        self.conv2d = nn.ConvTranspose2d(64, 64, kernel_size=4, padding=1, stride=2)
        self.conv3d = nn.ConvTranspose2d(64, 1, kernel_size=4, padding=1, stride=2)

        self.sigmoid = nn.Sigmoid()
        self.iterations = 0

        self.init_weights()

    def encode(self, x):
        x = x.view(-1,1,32,32)
        batch_size = x.shape[0]
        x = F.selu(self.conv1e(x))
        x = F.selu(self.conv2e(x))
        x = F.selu(self.conv3e(x))
        h1 = x.view(batch_size, -1)
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def decode(self, z, c):
        # c is the class
        c = idx2onehot(c, 10).to(self.device)
        z = torch.cat((z, c), dim=-1)

        h3 = F.selu(self.fc3(z)).view(-1,64,4,4)
        x = F.selu(self.conv1d(h3))
        x = F.selu(self.conv2d(x))
        x = self.conv3d(x)
        return self.sigmoid(x.view(-1, 32*32))

    def forward(self, x, c):
        mu, logvar = self.encode(x.view(-1, 32*32))
        z = self.reparameterize(mu, logvar)
        return self.decode(z, c), mu, logvar

    def sample(self, n, z=None, c=None):
        if z is None:
            z = Variable(torch.randn(n, self.lat_dim)).to(self.device)
        z = z.to(self.device)
        if c is None:
            c = torch.LongTensor(np.random.randint(0, 9, n)).view(-1).to(self.device)
        sample = self.decode(z, c)
        return sample, c

    def init_weights(self):
        self.apply(self.init_weights_dist)

    def init_weights_dist(self, m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform(m.weight)
            m.bias.data.fill_(0.01)
        if type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
            torch.nn.init.xavier_uniform(m.weight)