from __future__ import print_function
import numpy as np
import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from IPython import embed

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


class Net(nn.Module):
    def __init__(self):
        # Number of iterations we trained for
        super(Net, self).__init__()
        self.reset_net_attributes()

    def reset_net_attributes(self):
        self.curr_iteration = 0
        # List with the validation errors on the two splits
        self.valid_err_split_1 = []
        self.valid_err_split_2 = []
        self.valid_err_split_full = []
        # List with the iterations where we computed validation errors on the two splits
        self.valid_err_iterations = []

# Define classifier networks. Theta start will be trained on everything, theta all only on the restricted set
class _classifier(Net):
    def __init__(self):
        super(_classifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3)
        self.fc1 = nn.Linear(64, 50)
        self.fc2 = nn.Linear(50, 10)
        self.activ = F.selu

    def forward(self, x):
        x = x.view(-1, 1, 32, 32)
        x = self.activ(F.max_pool2d(self.conv1(x), 2))
        x = self.activ(F.max_pool2d(self.conv2(x), 2))
        x = self.activ(F.max_pool2d(self.conv3(x), 2))
        x = x.view(-1, 64)
        x = self.activ(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

    def init_weights(self):
        self.apply(self.init_weights)

    def init_weights_dist(self, m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)
        if type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
            torch.nn.init.xavier_uniform_(m.weight)

    def reset_net_attributes(self):
        super(_classifier, self).reset_net_attributes()

class VAE(Net):
    def __init__(self, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
                 lat_dim=20, architecture='standard', activ=F.selu, color_channels=None):
        """architecture: 'standard' use the standard VAE, 'large' use a large decoder, 'small' use a small decoder"""
        super(VAE, self).__init__()
        self.lat_dim = lat_dim
        self.device = device
        self.num_classes = 10
        self.architecture = architecture
        self.sigmoid = nn.Sigmoid()
        self.iterations = 0
        self.activ = activ
        self.color_channels = color_channels
        self.filters = 64
        if self.color_channels == 3:
            self.filters = 256


        #Encoder
        self.conv1e = nn.Conv2d(self.color_channels,  self.filters, kernel_size=4, stride=2)
        self.conv2e = nn.Conv2d(self.filters,  self.filters, kernel_size=4, stride=2)
        self.conv3e = nn.Conv2d(self.filters,  64, kernel_size=4, stride=2)
        self.fc21 = nn.Linear(256, self.lat_dim)
        self.fc22 = nn.Linear(256, self.lat_dim)

        if architecture == 'standard':
            self.standard_decoder()
        elif architecture == 'large':
            self.large_decoder()
        elif architecture == 'small':
            self.small_decoder()
        elif architecture == 'fully_connected':
            self.fully_connected_decoder()

        self.init_weights()

    def standard_decoder(self):
        #Decoder
        self.fc3 = nn.Linear(self.lat_dim + self.num_classes, 64*4*4)
        self.conv1d = nn.ConvTranspose2d(64, self.filters, kernel_size=4, padding=1, stride=2)
        self.conv2d = nn.ConvTranspose2d(self.filters, self.filters, kernel_size=4, padding=1, stride=2)
        self.conv3d = nn.ConvTranspose2d(self.filters, self.color_channels, kernel_size=4, padding=1, stride=2)

    def small_decoder(self):
        #Decoder
        self.fc3 = nn.Linear(self.lat_dim + self.num_classes, 64*4*4)
        self.conv1d = nn.ConvTranspose2d(64, self.filters//2, kernel_size=4, padding=1, stride=2)
        self.conv2d = nn.ConvTranspose2d(self.filters//2, self.filters//2, kernel_size=4, padding=1, stride=2)
        self.conv3d = nn.ConvTranspose2d(self.filters//2, self.color_channels, kernel_size=4, padding=1, stride=2)


    def large_decoder(self):
        #Decoder
        self.fc3 = nn.Linear(self.lat_dim + self.num_classes, 64*4*4)
        self.conv1d = nn.ConvTranspose2d(64, self.filters*2, kernel_size=4, padding=1, stride=2)
        self.conv2d = nn.ConvTranspose2d(self.filters*2, self.filters*2, kernel_size=4, padding=1, stride=2)
        self.conv3d = nn.ConvTranspose2d(self.filters*2, self.color_channels, kernel_size=4, padding=1, stride=2)


    def fully_connected_decoder(self):
        #Decoder
        self.fc3 = nn.Linear(self.lat_dim + self.num_classes, 64*4*4)
        self.fc4 = nn.Linear(64*4*4, 1024)
        self.fc5 = nn.Linear(1024, 32*32*self.color_channels)

    def encode(self, x):
        x = x.view(-1,self.color_channels,32,32)
        batch_size = x.shape[0]
        x = self.activ(self.conv1e(x))
        x = self.activ(self.conv2e(x))
        x = self.activ(self.conv3e(x))
        h1 = x.view(batch_size, -1)
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def decode_linear(self, z, c):
        # c is the class
        c = idx2onehot(c, 10).to(self.device)
        z = torch.cat((z, c), dim=-1)

        h3 = self.activ(self.fc3(z))
        x = self.activ(self.fc4(h3))
        x = self.fc5(x)
        return self.sigmoid(x.view(-1, 32*32*self.color_channels))

    def decode(self, z, c):
        if self.architecture == 'fully_connected':
            return self.decode_linear(z, c)
        # c is the class
        c = idx2onehot(c, 10).to(self.device)
        z = torch.cat((z, c), dim=-1)

        h3 = self.activ(self.fc3(z)).view(-1,64,4,4)
        x = self.activ(self.conv1d(h3))
        x = self.activ(self.conv2d(x))
        x = self.conv3d(x)
        return self.sigmoid(x.view(-1, 32*32*self.color_channels))

    def forward(self, x, c):
        mu, logvar = self.encode(x.view(-1, 32*32*self.color_channels))
        z = self.reparameterize(mu, logvar)
        return self.decode(z, c), mu, logvar

    def sample(self, n, z=None, c=None):
        if z is None:
            z = Variable(torch.randn(n, self.lat_dim)).to(self.device)
        z = z.to(self.device)
        if c is None:
            c = torch.LongTensor(np.random.randint(0, 9, n)).view(-1).to(self.device)
        sample = self.decode(z, c)
        return sample, c, z

    def init_weights(self):
        self.apply(self.init_weights_dist)

    def init_weights_dist(self, m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)
        if type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
            torch.nn.init.xavier_uniform_(m.weight)

    def reset_net_attributes(self):
        super(VAE, self).reset_net_attributes()