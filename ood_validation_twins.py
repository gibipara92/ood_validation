from __future__ import print_function
import argparse
import os
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
from models import VAE, idx2onehot

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

# custom weights initialization
def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)


# TODO: make this class a parent of both _classifier and VAE, so we can have common attributes for both.
# TODO: Probably need to move this and the classifier to the models.py file
class net(nn.Module):
    def __init__(self):
        # Number of iterations we trained for
        self.curr_iteration = 0
        # List with the validation errors on the two splits
        self.valid_err_split_1 = []
        self.valid_err_split_2 = []
        # List with the iterations where we computed validation errors on the two splits
        self.valid_err_iterations = []


# Define classifier networks. Theta start will be trained on everything, theta all only on the restricted set
class _classifier(nn.Module):
    def __init__(self):
        super(_classifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

    def init_weights(self):
        self.apply(self.init_weights)

    def init_weights_dist(self, m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform(m.weight)
            m.bias.data.fill_(0.01)
        if type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
            torch.nn.init.xavier_uniform(m.weight)

# TODO define a new class NET that has the extra attributes we need (e.g. validation scores, etc.)



# fixed_x_train_non_iid, fixed_y_train_non_iid = torch.Tensor(
#     ground_truth.sample(x_range=non_iid_training_range, n=1024, sample_grid=True)).to(device)
# fixed_x_valid_non_iid, fixed_y_valid_non_iid = torch.Tensor(
#     ground_truth.sample(x_range=validation_range, n=1024, sample_grid=True)).to(device)
# fixed_x_iid, fixed_y_iid = torch.Tensor(ground_truth.sample(x_range=iid_available_range, n=1024, sample_grid=True)).to(
#     device)
# fixed_x_test_non_iid, fixed_y_test_non_iid = torch.Tensor(
#     ground_truth.sample(x_range=test_range, n=1024, sample_grid=True)).to(device)
#
# fixed_x_train_non_iid_numpy, fixed_y_train_non_iid_numpy = fixed_x_train_non_iid.cpu().numpy(), fixed_y_train_non_iid.cpu().numpy()
# fixed_x_valid_non_iid_numpy, fixed_y_valid_non_iid_numpy = fixed_x_valid_non_iid.cpu().numpy(), fixed_y_valid_non_iid.cpu().numpy()
# fixed_x_iid_numpy, fixed_y_iid_numpy = fixed_x_iid.cpu().numpy(), fixed_y_iid.cpu().numpy()
# fixed_x_test_non_iid_numpy, fixed_y_test_non_iid_numpy = fixed_x_test_non_iid.cpu().numpy(), fixed_y_test_non_iid.cpu().numpy()


# Comments:
# Do it for both classification and standard retraining of the CVAE! Must be general

# The split on the Gaussian should be done by using a line. i.e., randomely initialize dim(z) vectors of size z each,
# and use each one of them as a point to compute a plane split. Use signum of dot product to check where it belongs,
# then flip the sign if wrong side.

# Do we keep the Gaussian term when retraining? Probably not

# List of needed for later:

# Data from split 1 for testing
# Data from split 2 for testing
# Data from outside of restricted area
# A function that given params trains a network with that configuration, will use this A LOT
# How to properly compare split 1 and 2 so we have full visibility?


################################
# Helper functions
################################

def train_classifier(net, distribution, iterations):
    """Trains net as a classifier on the distribution specified by distribution"""
    for it in range(iterations):

        # while training keep track of validation performances on all splits
        valid_err_split_1, valid_err_split_2 = validate_all(net)

    pass

def validate_all(net, append_results=True):
    """Returns validation scores for net on the given distribution
    Returns validation scores first on split_1, then split_2
    """
    valid_err_split_1 = 0 # TODO, use some standard fixed data from the two splits to compute this
    valid_err_split_2 = 0 # TODO
    if append_results:
        net.valid_err_split_1.append(valid_err_split_1)
        net.valid_err_split_2.append(valid_err_split_2)
        net.valid_err_iterations.append(net.curr_iteration)
    return valid_err_split_1, valid_err_split_2


def train_generator(net, distribution, iterations):
    """Trains net as a generator on the distribution specified by distribution"""
    pass


def train_twins(net, iterations):
    """Clones net, trains it on two splits separately as twins, compares the result"""
    net_twin_1 = net.clone()
    net_twin_2 = net.clone()

    # TODO we should be comparing the two nets all the time while training, not only at the end of training
    # TODO so that we can plot how the two twins get 'back together' for the nice case as we train
    if opt.task == 'classification':
        # Train theta on the two sets of the data
        train_classifier(net_twin_1, distribution=split_1, iterations=iterations)
        train_classifier(net_twin_2, distribution=split_2, iterations=iterations)
        compare_networks(net_twin_1, net_twin_2, data=restrict_gaussian_range)
    else:
        pass


def compare_networks(net1, net2, data):
    """ Compare performance of net1 and net2 on a given set of data, by comparing the average error between the
    predictions of the two. Returns the average cross error"""
    # TODO: data will be the range of the distrivution to use?
    net1.eval()
    net2.eval()
    out1 = net1(data)
    out2 = net2(data)
    # TODO: error = F.mse(out1, out2)
    # Print error


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

# Train the cvae if not already trained, otherwise load it
data_generator = torch.load(torch.load(opt.c_vae_path))
# data_generator.load_state_dict(torch.load(opt.c_vae_path))
data_generator.eval()

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

    train_twins(theta_star_eps, iterations=100)

    train_twins(theta_all_eps, iterations=100)

writer.close()
