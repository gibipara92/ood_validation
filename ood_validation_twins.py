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
from models import VAE, idx2onehot, Net, _classifier

# Args pars
if 1:
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='celeba',
                        help='cifar10 | lsun | imagenet | folder | lfw | fake | celeba | mnist')
    parser.add_argument('--task', default='classification', help='classification | generation')
    parser.add_argument('--dataroot', help='path to dataset')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
    parser.add_argument('--batch_size', type=int, default=1024, help='input batch size')
    parser.add_argument('--niter', type=int, default=5000, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.005, help='learning rate, default=0.0002')
    parser.add_argument('--beta', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--cuda', action='store_true', default=True, help='enables cuda')
    parser.add_argument('--debug', action='store_true', default=False, help='debug mode')
    parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
    parser.add_argument('--outf',
                        default='/is/cluster/gparascandolo/ood_validation/generator/',
                        help='folder to output images and model checkpoints')
    parser.add_argument('--c_vae_path',
                        default='/is/cluster/gparascandolo/generative-models/GAN/R_out/VAEs/CVAE/CVAE.pth',
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
    def __init__(self, net=torch.load(opt.c_vae_path, map_location=None if device is 'cuda' else 'cpu')):
        # Number of iterations we trained for
        self.net = VAE(device)
        self.net.load_state_dict(torch.load(opt.c_vae_path))
        self.net.to(device)
        self.net.eval()

    def sample_distribution(self, n, distribution, c=None):
        self.net.eval()
        # Improve this sampling cut to a nicer line
        if distribution == 'all':
            # This samples from the whole gaussian
            z = Variable(torch.randn(n, self.net.lat_dim)).to(self.net.device)
        else:
            # This restricts the value to the hypercube of side length 1.0
            # TODO implement trucnated normal instead of 0.01
            z = Variable(torch.randn(n, self.net.lat_dim)).to(self.net.device)

            if distribution is split_1:
                z[:,:15] = torch.abs(z[:,:15])
            elif distribution is split_2:
                z[:,:15] = -torch.abs(z[:,:15])
            elif distribution is restrict_gaussian_range:
                z[:,:15] = torch.abs(z[:,:15])
                z[n//2:,:15] = -z[n//2:,:15]
            else:
                print('Error in sample distribution, unknown distribution:', distribution)
                sys.exit(1)

        return self.sample(n, z, c)

    def sample(self, n, z=None, c=None):
        with torch.no_grad():
            return self.net.sample(n, z, c)


# Train the cvae if not already trained, otherwise load it
data_generator = Ground_truth()
# data_generator.load_state_dict(torch.load(opt.c_vae_path))

# Creat fixed sets of test data for the two splits
split_1_test_img, split_1_test_c, split_1_test_z = data_generator.sample_distribution(n=512, distribution=split_1)
split_2_test_img, split_2_test_c, split_2_test_z = data_generator.sample_distribution(n=512, distribution=split_2)
# Fixed distribution from the restricted range
test_both_splits_img, test_both_splits_c, test_both_splits_z = torch.cat((split_1_test_img[:256], split_2_test_img[:256])),\
                                                               torch.cat((split_1_test_c[:256], split_2_test_c[:256])), \
                                                               torch.cat((split_1_test_z[:256], split_2_test_z[:256]))
# Fixed distribution from the full gaussian
test_full_img, test_full_c, test_full_z = data_generator.sample_distribution(n=512, distribution='all')

# Comments:

# The split on the Gaussian should be done by using a line. i.e., randomely initialize dim(z) vectors of size z each,
# and use each one of them as a point to compute a plane split. Use signum of dot product to check where it belongs,
# then flip the sign if wrong side.

# Do we keep the Gaussian term when retraining? Probably not

################################
# Helper functions
################################

def clone_network(net):
    if opt.task == 'classification':
        return copy.deepcopy(net)
    else:
        net.device = None
        copy_net = copy.deepcopy(net)
        copy_net.device = device
        return copy_net


def _optimizer(net, lr=0.0001):
    """Returns an optimizer for the given network"""
    return optim.Adam(net.parameters(), lr=lr)


def train_generator(net, distribution, iterations, optimizer=None):
    """Trains net as a generator on the distribution specified by distribution"""
    if optimizer is None:
        optimizer = _optimizer(net)

    it = 0

    valid_err_split_1, valid_err_split_2 = validate_all(net, append_results=True)
    while it < iterations:
        net.train()
        data, c, z = data_generator.sample_distribution(n=opt.batch_size, distribution=distribution) #
        it += 1
        net.curr_iteration += 1
        data, c, z = data.to(device), c.to(device), z.to(device)
        optimizer.zero_grad()
        output = net.decode(z, c)
        loss = F.binary_cross_entropy(output.view(-1, 32*32), data)
        loss.backward()
        optimizer.step()
        if net.curr_iteration % 100 == 0:
            valid_err_split_1, valid_err_split_2 = validate_all(net, append_results=True)
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                net.curr_iteration, net.curr_iteration * len(data), net.curr_iteration,
                       100. * net.curr_iteration / net.curr_iteration, loss.item()))

def train_classifier(net, distribution, iterations, optimizer=None):
    """Trains net as a classifier on the distribution specified by distribution.
    If optimizer is None it will initialize a new one."""
    if optimizer is None:
        optimizer = _optimizer(net)

    it = 0

    valid_err_split_1, valid_err_split_2 = validate_all(net, append_results=True)
    while it < iterations:
        net.train()
        data, target, _ = data_generator.sample_distribution(n=opt.batch_size, distribution=distribution)
        it += 1
        net.curr_iteration += 1
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = net(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if net.curr_iteration % 100 == 0:
            valid_err_split_1, valid_err_split_2 = validate_all(net, append_results=True)
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                net.curr_iteration, net.curr_iteration * len(data), net.curr_iteration,
                       100. * net.curr_iteration / net.curr_iteration, loss.item()))


def validation_err_on_data(net, data, targets):
    return F.nll_loss(net(data), targets).item()


def validation_err_on_data_generator(net, data, c, z):
    return F.binary_cross_entropy(net.decode(z, c).view(-1, 32*32), data).item()


def validate_all(net, append_results=True, task=opt.task):
    """Returns validation scores for net on the given distribution
    Returns validation scores first on split_1, then split_2
    """
    net.eval()
    with torch.no_grad():
        if task == 'classification':
            valid_err_split_1 = validation_err_on_data(net, split_1_test_img, split_1_test_c)
            valid_err_split_2 = validation_err_on_data(net, split_2_test_img, split_2_test_c)
        else:
            valid_err_split_1 = validation_err_on_data_generator(net, split_1_test_img, split_1_test_c, split_1_test_z)
            valid_err_split_2 = validation_err_on_data_generator(net, split_2_test_img, split_2_test_c, split_2_test_z)

        if append_results:
            net.valid_err_split_1.append(valid_err_split_1)
            net.valid_err_split_2.append(valid_err_split_2)
            net.valid_err_iterations.append(net.curr_iteration)

        return valid_err_split_1, valid_err_split_2


def train_twins(net, iterations, tensorboard_id=None):
    """Clones net, trains it on two splits separately as twins, compares the result"""

    net_twin_1 = clone_network(net)
    net_twin_2 = clone_network(net)

    optim_1 = _optimizer(net_twin_1)
    optim_2 = _optimizer(net_twin_2)

    # Will store here the disagreemnet between the two twins
    disagreement_error = []

    disagreement_error.append(

    compare_networks(net_twin_1, net_twin_2, data=test_both_splits_img, z=test_both_splits_z, c=test_both_splits_c))
    if opt.task == 'classification':
        train_net = train_classifier
    else:
        train_net = train_generator

    for i in range(40):
        # Train theta on the two sets of the data
        train_net(net_twin_1, distribution=split_1, optimizer=optim_1, iterations=iterations)
        train_net(net_twin_2, distribution=split_2, optimizer=optim_2, iterations=iterations)
        disagreement_error.append(compare_networks(net_twin_1, net_twin_2, data=test_both_splits_img, z=test_both_splits_z, c=test_both_splits_c))
        if tensorboard_id is not None:
            writer.add_scalar(tensorboard_id + '/disagreement_error', disagreement_error[-1], net_twin_1.curr_iteration)

    plt.close()
    plt.figure()
    plt.gca().set_title('Twin 1 on splits 1 and 2')
    plt.plot(net_twin_1.valid_err_iterations, net_twin_1.valid_err_split_1, label='1 on split 1')
    plt.plot(net_twin_1.valid_err_iterations, net_twin_1.valid_err_split_2, label='1 on split 2')
    plt.legend()
    plt.savefig(opt.outf + 'twins_1' + str(tensorboard_id).replace('/', '_') + '.png')

    plt.close()
    plt.figure()
    plt.gca().set_title('Twin 2 on splits 1 and 2')
    plt.plot(net_twin_2.valid_err_iterations, net_twin_2.valid_err_split_1, label='2 on split 1')
    plt.plot(net_twin_2.valid_err_iterations, net_twin_2.valid_err_split_2, label='2 on split 2')
    plt.legend()
    plt.savefig(opt.outf + 'twins_2' + str(tensorboard_id).replace('/', '_') + '.png')

    return disagreement_error


def compare_networks(net1, net2, data=None, c=None, z=None):
    """ Compare performance of net1 and net2 on a given set of data, by comparing the average error between the
    predictions of the two. Returns the average cross error"""
    with torch.no_grad():
        net1.eval()
        net2.eval()

        if opt.task == 'classification':
            out1 = net1(data)
            out2 = net2(data)
            discrepancy_error = F.mse_loss(torch.exp(out1), torch.exp(out2), size_average=True)  # TODO, uniform losses in the script (now mix of mse and log)
        else:
            out1 = net1.decode(z, c)
            out2 = net2.decode(z, c)
            discrepancy_error = F.mse_loss(out1, out2, size_average=True)

        return discrepancy_error.item()


def add_noise_weights(net, eps_sigma):
    """ Add gaussian noise to the weights of net as scaled by eps. Modify in place, no return"""
    if eps_sigma == 0.:
        return
    for m in net.parameters():
        m.data += torch.FloatTensor(m.size()).normal_(0., eps_sigma).to(device)



################################
### Main loop
################################


#########################
### Pre-setting
#########################

print("Pretraining theta star")
# Init and train theta_star on all of the data from the cvae
if opt.task == 'classification':
    theta_star = _classifier()
    theta_star.to(device)
    # Train theta star on *all* of the data
    optim_star = _optimizer(theta_star, lr=0.001)
    train_classifier(theta_star, distribution='all', iterations=10000 if not opt.debug else 100, optimizer=optim_star)
else:
    # For the VAE we don't retrain but simply load
    theta_star = VAE(device=None)
    theta_star.to(device)
    theta_star.device = device
    theta_star.load_state_dict(torch.load(opt.c_vae_path))

# Save the weights
torch.save(theta_star.state_dict(), '%s/theta_star.pth' % (opt.outf))

print("Pretraining theta all")
# Train theta_all on all of the restricted data from the cvae
if opt.task == 'classification':
    theta_all = _classifier()
    theta_all.to(device)
    # Train theta all on the restricted set of the data
    optim_all = _optimizer(theta_all, lr=0.001)
    train_classifier(theta_all, distribution=restrict_gaussian_range, iterations=10000 if not opt.debug else 100, optimizer=optim_all)
else:
    theta_all = VAE(device=None)
    theta_all.to(device)
    theta_all.device = device
    optim_all = _optimizer(theta_all, lr=0.001)
    train_generator(theta_all, distribution=restrict_gaussian_range, iterations=10000 if not opt.debug else 2000)

# Save the weights
torch.save(theta_all.state_dict(), '%s/theta_all.pth' % (opt.outf))

print("Comparing performance of theta star and theta all on the restricted set of data")
# Ensure that validation iid error on restricted is approx the same # TODO print this
compare_networks(theta_star, theta_all, data=test_both_splits_img, z=test_both_splits_z, c=test_both_splits_c)

## Main experiment: twin study
# for growing epsilon noise:
#       reload theta_star and theta_all
#       perturb both with noise epsilon
#       train 6 models:
#           star on 1, star on 2, star on 1 and 2
#           all on 1, all on 2, all on 1 and 2
#           keep track of *all* validation measures
#           keep track of *cross-error on the other splits*

print("Twin study")

epsilons_to_test = [i * 0.05 for i in range(2,4)]
disagreement_error_star = []
disagreement_error_all  = []

for i, eps in enumerate(epsilons_to_test):
    disagreement_error_star_local = []
    disagreement_error_all_local = []
    for _ in range(5 if not opt.debug else 2):
        print("Eps std", eps)

        # Clone the network
        theta_star_eps = clone_network(theta_star)
        # Reset all the counters and validation measures
        theta_star_eps.reset_net_attributes()
        # Add noise
        add_noise_weights(theta_star_eps, eps_sigma=eps)

        # Clone the network
        theta_all_eps = clone_network(theta_all)
        # Reset all the counters and validation measures
        theta_all_eps.reset_net_attributes()
        # Add noise
        add_noise_weights(theta_all_eps, eps_sigma=eps)

        print("Train star twins")
        disagreement_error_star_local.append(train_twins(theta_star_eps, iterations=400 if not opt.debug else 40, tensorboard_id='eps_std{:.4}/Star'.format(eps)))
        print("Train all twins")
        disagreement_error_all_local.append(train_twins(theta_all_eps, iterations=400 if not opt.debug else 40, tensorboard_id='eps_std{:.4}/All'.format(eps)))

    disagreement_error_star.append(np.mean(disagreement_error_star_local, 0))
    disagreement_error_all.append(np.mean(disagreement_error_all_local, 0))

    star = disagreement_error_star
    all = disagreement_error_all

    star = np.array(star)
    all  = np.array(all)

    plt.close()
    plt.title('Twin disagreement')
    plt.xlabel('Iterations')
    plt.xlabel('Disagreement loss')
    plt.plot(star[i].T, c='b')
    plt.plot(all[i].T, c='r')
    plt.legend()
    plt.savefig(opt.outf + str(epsilons_to_test[i])+'.png')

writer.close()
