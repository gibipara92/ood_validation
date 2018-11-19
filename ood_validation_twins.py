from __future__ import print_function
import argparse
import os, sys
import random
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from IPython import embed
import scipy.misc
import numpy as np
import matplotlib

matplotlib.use('agg')
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
from models import VAE, idx2onehot, net, _classifier

# Args pars
if 1:
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='celeba',
                        help='cifar10 | lsun | imagenet | folder | lfw | fake | celeba | mnist')
    parser.add_argument('--task', default='classification', help='classification | generation')
    parser.add_argument('--dataroot', help='path to dataset')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
    parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
    parser.add_argument('--niter', type=int, default=5000, help='number of epochs to train for')
    parser.add_argument('--input_size', type=int, default=1, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.005, help='learning rate, default=0.0002')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--sigma', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--train_sigma', type=float, default=1.0,
                        help='sigma for training distribution of full. If 1, train=test. For < 1 missing support')
    parser.add_argument('--cuda', action='store_true', default=True, help='enables cuda')
    parser.add_argument('--dcgan', action='store_true', default=False, help='enables cuda')
    parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
    parser.add_argument('--outf',
                        default='/is/cluster/gparascandolo/generative-models/GAN/R_out/ood_validation_twins/test/',
                        help='folder to output images and model checkpoints')
    parser.add_argument('--c_vae_path',
                        default='/is/cluster/gparascandolo/generative-models/GAN/R_out/VAEs/CVAE/CVAE.t7',
                        help='folder to output images and model checkpoints')
    parser.add_argument('--manualSeed', type=int, help='manual seed')

    opt = parser.parse_args()
    print(opt)

# Cuda and tensorboardX
if 1:
    from datetime import datetime

    now = datetime.now()
    tbdir = now.strftime("%Y%m%d-%H%M%S") + "/"

    try:
        os.makedirs(opt.outf)
    except OSError:
        pass

    try:
        os.makedirs(opt.outf + '/runs/' + tbdir)
    except OSError:
        pass
    writer = SummaryWriter(opt.outf + '/runs/' + tbdir)

    if opt.manualSeed is None:
        opt.manualSeed = random.randint(1, 10000)
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    if opt.cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(opt.manualSeed)
        cudnn.benchmark = True
    else:
        opt.cuda = False

    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


########################################################################################################################
#                  STARTS HERE
########################################################################################################################

# Specify ranges for available data, validation data in non-iid, and unseen test data
restrict_gaussian_range = [-1, 1]
split_1 = [-1, 0]
split_2 = [0, 1]


ngpu = int(opt.ngpu)
nc = 3
if opt.dataset in ['mnist']:
    nc = 1

# # custom weights initialization
# def init_weights(m):
#     if type(m) == nn.Linear:
#         torch.nn.init.xavier_uniform(m.weight)
#         m.bias.data.fill_(0.01)


class Ground_truth():
    """ For now it only loads the c_vae_path as ground truth"""
    def __init__(self, net=torch.load(opt.c_vae_path)):
        # Number of iterations we trained for
        self.net = torch.load(net)
        self.net.eval()

    def sample_distribution(self, n, distribution, c=None):
        # Improve this sampling cut to a nicer line
        if distribution == 'all':
            # This samples from the whole gaussian
            z = Variable(torch.randn(n, self.net.lat_dim)).to(self.device)
        else:
            # This restricts the value to the hypercube of side length 1.0
            z = Variable(0.5 - torch.rand(n, self.net.lat_dim)).to(self.device)

        if distribution is split_1:
            z[:,0] = torch.abs(z[:,0])
        elif distribution is split_2:
            z[:,0] = -torch.abs(z[:,0])
        elif distribution is restrict_gaussian_range:
            # No need to do anything in this case, will be in the hypercube
            pass
        else:
            print('Error in sample distribution, unknown distribution:', distribution)
            sys.exit(1)

        return self.sample(n, z, c)

    def sample(self, n, z=None, c=None):
        return self.net.sample(n, z, c)


# Train the cvae if not already trained, otherwise load it
data_generator = Ground_truth()
# data_generator.load_state_dict(torch.load(opt.c_vae_path))

# Creat fixed sets of test data for the two splits
split_1_test_img, split_1_test_c = data_generator.sample_distribution(n=512, distribution=split_1)
split_2_test_img, split_2_test_c = data_generator.sample_distribution(n=512, distribution=split_2)
# Fixed distribution from the restricted range
test_both_splits_img, test_both_splits_c = torch.cat(split_1_test_img[:256], split_2_test_img[:256]),\
                                           torch.cat(split_1_test_c[:256], split_2_test_c[:256])
# Fixed distribution from the full gaussian
test_full_img, test_full_c = data_generator.sample_distribution(n=512, distribution='all')


# Comments:

# The split on the Gaussian should be done by using a line. i.e., randomely initialize dim(z) vectors of size z each,
# and use each one of them as a point to compute a plane split. Use signum of dot product to check where it belongs,
# then flip the sign if wrong side.

# Do we keep the Gaussian term when retraining? Probably not

################################
# Helper functions
################################


def _optimizer(net):
    """Returns an optimizer for the given network"""
    return optim.RMSprop(net.parameters(), lr=0.01)


def train_generator(net, distribution, iterations, optimizer=None): # TODO (steal from train cvae file and train_classifier)
    """Trains net as a generator on the distribution specified by distribution"""
    pass


def train_classifier(net, distribution, iterations, optimizer=None):
    """Trains net as a classifier on the distribution specified by distribution.
    If optimizer is None it will initialize a new one."""

    if optimizer is None:
        optimizer = _optimizer(net)

    net.train()
    it = 0
    while it < iterations:
        data, target = data_generator.sample_distribution(n=256, distribution=distribution)
        it += 1
        net.curr_iteration += 1
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = net(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if net.curr_iteration % opt.log_interval == 0:
            valid_err_split_1, valid_err_split_2 = validate_all(net, append_results=True)
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                net.iterations, net.curr_iteration * len(data), net.curr_iteration,
                       100. * net.curr_iteration / net.curr_iteration, loss.item()))


def validation_err_on_data(net, data, targets):
    return F.mse_loss(net(data), targets)


def validate_all(net, append_results=True):
    """Returns validation scores for net on the given distribution
    Returns validation scores first on split_1, then split_2
    """
    net.eval()

    valid_err_split_1 = validation_err_on_data(net, split_1_test_img, split_1_test_c)
    valid_err_split_2 = validation_err_on_data(net, split_2_test_img, split_2_test_c)

    if append_results:
        net.valid_err_split_1.append(valid_err_split_1)
        net.valid_err_split_2.append(valid_err_split_2)
        net.valid_err_iterations.append(net.curr_iteration)

    return valid_err_split_1, valid_err_split_2


def train_twins(net, iterations):
    """Clones net, trains it on two splits separately as twins, compares the result"""
    net_twin_1 = net.clone()
    net_twin_2 = net.clone()

    optim_1 = _optimizer(net_twin_1)
    optim_2 = _optimizer(net_twin_2)

    # Will store here the disagreemnet between the two twins
    disagreement_error = []

    if opt.task == 'classification':
        disagreement_error.append([net_twin_1.curr_iteration, compare_networks(net_twin_1, net_twin_2, data=restrict_gaussian_range)])
        for i in range(40):
            # Train theta on the two sets of the data
            train_classifier(net_twin_1, distribution=split_1, optimizer=optim_1, iterations=iterations)
            train_classifier(net_twin_2, distribution=split_2, optimizer=optim_2, iterations=iterations)
            disagreement_error.append(compare_networks(net_twin_1, net_twin_2, data=restrict_gaussian_range))
    else:
        pass
    return disagreement_error


def compare_networks(net1, net2, data):
    """ Compare performance of net1 and net2 on a given set of data, by comparing the average error between the
    predictions of the two. Returns the average cross error"""
    net1.eval()
    net2.eval()

    out1 = net1(data)
    out2 = net2(data)

    discrepancy_error = F.mse_loss(out1, out2, size_average=True)
    return discrepancy_error


def add_noise_weights(net, eps_sigma):
    """ Add gaussian noise to the weights of net as scaled by eps. Modify in place, no return"""
    for m in net.parameters():
        m.data += torch.FloatTensor(m.size()).normal_(0., eps_sigma).to(device)



################################
### Main loop
################################


#########################
### Pre-setting
#########################


# Init and train theta_star on all of the data from the cvae
if opt.task == 'classification':
    theta_star = _classifier()
    theta_star.to(device)
    # Train theta star on *all* of the data
    train_classifier(theta_star, distribution='all', iterations=100)
else:
    # For the VAE we don't retrain but simply load
    theta_star = VAE(device=device)
    theta_star.load_state_dict(torch.load(opt.c_vae_path))
    theta_star.to(device)

# Save the weights
torch.save(theta_star, '%s/theta_star.t7' % (opt.outf))

# Train theta_all on all of the restricted data from the cvae
if opt.task == 'classification':
    theta_all = _classifier()
    theta_all.to(device)
    # Train theta all on the restricted set of the data
    train_classifier(theta_all, distribution=restrict_gaussian_range, iterations=100)
else:
    theta_all = VAE(device=device)
    theta_all.to(device)
    train_generator(theta_all, distribution=restrict_gaussian_range, iterations=100)

# Save the weights
torch.save(theta_all, '%s/theta_all.t7' % (opt.outf))

# Ensure that validation iid error on restricted is approx the same
compare_networks(theta_star, theta_all, data=None)  # TODO fix the data to restricted test

## Main experiment: twin study
# for growing epsilon noise:
#       reload theta_star and theta_all
#       perturb both with noise epsilon
#       train 6 models:
#           star on 1, star on 2, star on 1 and 2
#           all on 1, all on 2, all on 1 and 2
#           keep track of *all* validation measures
#           keep track of *cross-error on the other splits*


epsilons_to_test = [i * 0.1 for i in range(5)]

for eps in epsilons_to_test:
    theta_star_eps = add_noise_weights(theta_star.clone())
    theta_all_eps  = add_noise_weights(theta_all.clone())

    disagreement_error_star = train_twins(theta_star_eps, iterations=100)
    disagreement_error_all  = train_twins(theta_all_eps, iterations=100)

    embed()

writer.close()
