from __future__ import print_function
import argparse, os
import numpy as np
import torch
import torch.utils.data
import torch.backends.cudnn as cudnn
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from models import VAE, idx2onehot
from IPython import embed

parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 132)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--path_img',   default='/is/cluster/gparascandolo/generative-models/GAN/R_out/VAEs/CVAE/',
                    help='Path', type=str)
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
args.lat_dim = 20

if not os.path.exists(args.path_img + ''):
    os.makedirs(args.path_img + '')

if args.cuda and torch.cuda.is_available():
    # torch.cuda.manual_seed_all(args.manualSeed)
    cudnn.benchmark = True
else:
    args.cuda = False

if torch.cuda.is_available() and not args.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST(root='/is/cluster/gparascandolo/mnist/', train=True, download=False,
                   transform=transforms.Compose([transforms.Pad(padding=2),
                                                 transforms.ToTensor()])),
                   batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST(root='/is/cluster/gparascandolo/mnist/', train=False, transform=transforms.Compose([transforms.Pad(padding=2),transforms.ToTensor()])),
                   batch_size=args.batch_size, shuffle=True, **kwargs)
num_classes = 10


# def idx2onehot(idx, n):
#     assert idx.size(1) == 1
#     assert torch.max(idx).data[0] < n
#     onehot = torch.zeros(idx.size(0), n)
#     onehot.scatter_(1, idx.data, 1)
#     onehot = torch.Tensor(onehot)
#     return onehot

model = VAE(device=device)
model.to(device)


def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 32*32))

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    # Normalise by same number of elements as in reconstruction
    KLD /= args.batch_size * 32*32

    return BCE + KLD


optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10000, gamma=0.2)


def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, targets) in enumerate(train_loader):
        model.iterations += 1
        scheduler.step()
        data = Variable(data)
        data = data.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data, targets)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.data[0]
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)], It: {}\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), model.iterations,
                loss.data[0] / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))


def test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, targets) in enumerate(test_loader):
            data = data.to(device)
            targets = targets.to(device)
            recon_batch, mu, logvar = model(data, targets)
            test_loss += loss_function(recon_batch, data, mu, logvar).item()
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n], recon_batch.view(args.batch_size, 1, 32, 32)[:n]])
                save_image(comparison.data.cpu(),
                             args.path_img + 'reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))


for epoch in range(1, args.epochs + 1):
    train(epoch)
    test(epoch)
    # sample = Variable(torch.randn(64, args.lat_dim))
    # sample = sample.to(device)
    targets = torch.LongTensor([[i] * 8 for i in range(10)]).view(-1).to(device)
    # sample = model.decode(sample, targets).cpu()
    sample = model.sample(n=80, c=targets).cpu()
    save_image(sample.data.view(80, 1, 32, 32), args.path_img + 'sample_' + str(epoch) + '.png', nrow=10)

torch.save(model, '%s/CVAE.t7' % (args.path_img))


