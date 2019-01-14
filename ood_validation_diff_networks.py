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
from torchvision.utils import save_image
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
    parser.add_argument('--dims_to_half', type=int, default=10, help='dimensions to half in the gaussian')
    parser.add_argument('--niter', type=int, default=5000, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.005, help='learning rate, default=0.0002')
    parser.add_argument('--threshold', type=float, default=0.000001, help='learning rate, default=0.0002')
    parser.add_argument('--beta', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--cuda', action='store_true', default=True, help='enables cuda')
    parser.add_argument('--debug', action='store_true', default=False, help='debug mode')
    parser.add_argument('--repetitions', type=int, default=3, help='How many times to repeat an experiment')
    parser.add_argument('--no_train_all', action='store_true', default=False, help='Use pretrained all network')
    parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
    parser.add_argument('--outf',
                        default='/is/cluster/gparascandolo/ood_validation_diff_nets/celeba/',
                        help='folder to output images and model checkpoints')
    parser.add_argument('--c_vae_path',
                        default='/is/cluster/gparascandolo/ood_validation_diff_nets/VAEs/',
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

def sample_z_distribution(distribution, n, z_dim, device):
    if distribution == 'all':
        # This samples from the whole gaussian
        z = Variable(torch.randn(n, z_dim)).to(device)
    else:
        # This restricts the value to the hypercube of side length 1.0
        # TODO implement trucnated normal instead of 0.01
        z = Variable(torch.randn(n, z_dim)).to(device)
        dims_to_half = opt.dims_to_half
        if distribution is split_1:
            z = Variable(0.05 * torch.randn(n, z_dim)).to(device).clamp(-0.1, 0.1) # TODO .clamp
            # z[:, :dims_to_half] = torch.abs(z[:, :dims_to_half])
        elif distribution is split_2:
            z = Variable(0.2 * torch.randn(n, z_dim)).to(device).clamp(-0.3, 0.3)
            # z[:, :dims_to_half] = -torch.abs(z[:, :dims_to_half])
        elif distribution is restrict_gaussian_range:
            # z[:, :dims_to_half] = torch.abs(z[:, :dims_to_half])
            z[:n//2] = Variable(0.05 * torch.randn(n, z_dim)).to(device)[:n//2].clamp(-0.1, 0.1)
            z[n//2:] = Variable(0.2 * torch.randn(n, z_dim)).to(device)[n//2:].clamp(-0.3, 0.3)
            # z[n // 2:, :dims_to_half] = -z[n // 2:, :dims_to_half]
        else:
            print('Error in sample distribution, unknown distribution:', distribution)
            sys.exit(1)
    return z

class Ground_truth():
    """ For now it only loads the c_vae_path as ground truth"""

    def __init__(self, path, color_channels, map_location=None if device is 'cuda' else 'cpu'):
        # Number of iterations we trained for
        self.net = VAE(device, color_channels=color_channels)
        self.net.load_state_dict(torch.load(path))
        self.net.to(device)
        self.net.eval()

    def sample_distribution(self, n, distribution, c=None):
        self.net.eval()
        # Improve this sampling cut to a nicer line
        z = sample_z_distribution(distribution, n, self.net.lat_dim, self.net.device)
        return self.sample(n, z, c)

    def sample(self, n, z=None, c=None):
        with torch.no_grad():
            return self.net.sample(n, z, c)


# Train the cvae if not already trained, otherwise load it
data_generator = Ground_truth(path=opt.c_vae_path + 'CVAE_' + opt.dataset + '.pth', color_channels=nc)
# data_generator.load_state_dict(torch.load(opt.c_vae_path))

# Creat fixed sets of test data for the two splits
split_1_test_img, split_1_test_c, split_1_test_z = data_generator.sample_distribution(n=512, distribution=split_1)
split_2_test_img, split_2_test_c, split_2_test_z = data_generator.sample_distribution(n=512, distribution=split_2)
# Fixed distribution from the restricted range
test_both_splits_img, test_both_splits_c, test_both_splits_z = torch.cat(
    (split_1_test_img[:256], split_2_test_img[:256])), \
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

def pad_all_sublists(super_list):
    max_len = 0
    for l in super_list:
        if len(l) > max_len:
            max_len = len(l)
    for l in super_list:
        l += [l[-1]] * (max_len - len(l))
    return super_list


def clone_network(net):
    if opt.task == 'classification':
        return copy.deepcopy(net)
    else:
        net.device = None
        copy_net = copy.deepcopy(net)
        copy_net.device = device
        return copy_net


def _optimizer(net, lr=0.01):
    """Returns an optimizer for the given network"""
    return optim.Adam(net.parameters(), lr=lr)


def _scheduler(optimizer):
    """Returns an optimizer for the given network"""
    return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.2, verbose=True, cooldown=100, min_lr=1e-6, patience=100)


def train_generator(net, distribution, iterations, optimizer=None, scheduler=None, threshold=0.0):
    """Trains net as a generator on the distribution specified by distribution. Returns if the net has converged"""
    if optimizer is None:
        optimizer = _optimizer(net)
        scheduler = _scheduler(optimizer)

    it = 0

    valid_err_split_1, valid_err_split_2, valid_err_full = validate_all(net, append_results=True)
    while it < iterations:
        net.train()
        data, c, z = data_generator.sample_distribution(n=opt.batch_size, distribution=distribution)  #
        it += 1
        net.curr_iteration += 1
        data, c, z = data.to(device), c.to(device), z.to(device)
        optimizer.zero_grad()
        output = net.decode(z, c)
        # loss = F.binary_cross_entropy(output.view(-1, 32 * 32), data)
        loss = F.mse_loss(output.view(-1, 32 * 32 * nc), data)
        loss.backward()
        optimizer.step()
        if net.curr_iteration % 100 == 0:
            valid_err_split_1, valid_err_split_2, valid_err_full = validate_all(net, append_results=True)
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                net.curr_iteration, net.curr_iteration * len(data), net.curr_iteration,
                                    100. * net.curr_iteration / net.curr_iteration, loss.item()))
        scheduler.step(loss.item())
        if loss.item() < threshold:
            print('Stop training, loss is very low', loss.item())
            return True

    return False


def train_classifier(net, distribution, iterations, optimizer=None, scheduler=None):
    """Trains net as a classifier on the distribution specified by distribution.
    If optimizer is None it will initialize a new one."""
    if optimizer is None:
        optimizer = _optimizer(net)
        scheduler = _scheduler(optimizer)
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
            valid_err_split_1, valid_err_split_2, valid_err_full = validate_all(net, append_results=True)
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                net.curr_iteration, net.curr_iteration * len(data), net.curr_iteration,
                                    100. * net.curr_iteration / net.curr_iteration, loss.item()))
        scheduler.step(loss.item())


def validation_err_on_data(net, data, targets):
    return F.nll_loss(net(data), targets).item()


def validation_err_on_data_generator(net, data, c, z):
    return F.mse_loss(net.decode(z, c).view(-1, 32 * 32 * nc), data, reduction='elementwise_mean').item()


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
            valid_err_split_full = validation_err_on_data_generator(net, test_full_img, test_full_c, test_full_z)

        if append_results:
            net.valid_err_split_1.append(valid_err_split_1)
            net.valid_err_split_2.append(valid_err_split_2)
            net.valid_err_split_full.append(valid_err_split_full)
            net.valid_err_iterations.append(net.curr_iteration)

        return valid_err_split_1, valid_err_split_2, valid_err_split_full


def save_samples(net, distribution, id=''):
    targets = torch.LongTensor([[i] * 8 for i in range(10)]).view(-1).to(device)
    # z = sample_z_distribution(distribution=distribution, n=80, z_dim=net.lat_dim, device=device)
    if distribution == 'all':
        z = test_full_z[:80]
    elif distribution == restrict_gaussian_range:
        z = test_both_splits_z[:80]
    elif distribution == split_1:
        z = split_1_test_z[:80]
    elif distribution == split_2:
        z = split_2_test_z[:80]
    else:
        raise ValueError('save_samples got wrong value for distribution')
    with torch.no_grad():
        sample, _, _ = net.sample(n=80, z=z, c=targets)
        gt, _, _ = data_generator.sample(n=80, z=z, c=targets)
    if opt.cuda:
        sample = sample.cpu()
        gt = gt.cpu()
    images = sample.clone()
    images[::2] = sample[:40]
    images[1::2] = gt[:40]
    save_image(images.data.view(80, nc, 32, 32), opt.outf + 'sample_' + id + '.png', nrow=8)


def train_twins(net, iterations, tensorboard_id=None, first_exp=True):
    """Clones net, trains it on two splits separately as twins, compares the result"""

    net_twin_1 = clone_network(net)
    net_twin_2 = clone_network(net)

    optim_1 = _optimizer(net_twin_1)
    sched_1 = _scheduler(optim_1)
    optim_2 = _optimizer(net_twin_2)
    sched_2 = _scheduler(optim_2)

    # Will store here the disagreemnet between the two twins
    disagreement_error = []

    disagreement_error.append( \
        compare_networks(net_twin_1, net_twin_2, data=test_both_splits_img, z=test_both_splits_z, c=test_both_splits_c))
    if opt.task == 'classification':
        train_net = train_classifier
    else:
        train_net = train_generator

    converged1, converged2 = False, False

    for i in range(500):
        # Train theta on the two sets of the data
        if not converged1:
            converged1 = train_net(net_twin_1, distribution=split_1, optimizer=optim_1, scheduler=sched_1, iterations=iterations, threshold=opt.threshold)
            if first_exp:
                save_samples(net_twin_1, distribution='all', id=tensorboard_id.replace('/', '') + '_split1_it' + str(i))
        # Here twin two is trained on both at the same time
        if not converged2:
            converged2 = train_net(net_twin_2, distribution=restrict_gaussian_range, optimizer=optim_2, scheduler=sched_2, iterations=iterations, threshold=opt.threshold)
            if first_exp:
                save_samples(net_twin_2, distribution='all', id=tensorboard_id.replace('/','') + '_restricted_it' + str(i))

        disagreement_error.append(
            compare_networks(net_twin_1, net_twin_2, data=test_both_splits_img, z=test_both_splits_z,
                             c=test_both_splits_c))

        if tensorboard_id is not None:
            writer.add_scalar(tensorboard_id + '/disagreement_error', disagreement_error[-1], net_twin_1.curr_iteration)


    return disagreement_error, net_twin_1.valid_err_split_1, net_twin_1.valid_err_split_2, net_twin_1.valid_err_split_full


# plt.close()
# plt.figure()
# plt.gca().set_title('Twin 2 on splits 1 and 2')
# plt.plot(net2.valid_err_iterations, net2.valid_err_split_1, label='2 on split 1')
# plt.plot(net2.valid_err_iterations, net2.valid_err_split_2, label='2 on split 2')
# plt.legend()
# plt.savefig(opt.outf + 'twins_2' + str(tensorboard_id).replace('/', '_') + '.png')


def compare_networks(net1, net2, data=None, c=None, z=None):
    """ Compare performance of net1 and net2 on a given set of data, by comparing the average error between the
    predictions of the two. Returns the average cross error"""
    with torch.no_grad():
        net1.eval()
        net2.eval()

        if opt.task == 'classification':
            out1 = net1(data)
            out2 = net2(data)
            discrepancy_error = F.mse_loss(torch.exp(out1), torch.exp(out2),
                                           reduction='elementwise_mean')  # TODO, uniform losses in the script (now mix of mse and log)
        else:
            out1 = net1.decode(z, c)
            out2 = net2.decode(z, c)
            discrepancy_error = F.mse_loss(out1, out2, reduction='elementwise_mean')

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

# Init and train theta_star on all of the data from the cvae
print("Pretraining theta star")
if opt.task == 'classification':
    theta_star = _classifier()
    theta_star.to(device)
    # Train theta star on *all* of the data
    optim_star = _optimizer(theta_star, lr=0.01)
    sched_star = _scheduler(optim_star)
    train_classifier(theta_star, distribution='all', iterations=10000 if not opt.debug else 100, optimizer=optim_star, scheduler=sched_star)
else:
    # print("Skip pre-training of theta star, start from loaded ground truth")
    # For the VAE we don't retrain but simply load
    theta_star = VAE(device=None, architecture='standard', color_channels=nc)
    theta_star.to(device)
    theta_star.device = device
    # theta_star.load_state_dict(torch.load(opt.c_vae_path))
    # if opt.no_train_all:
    #     print('Load pre-trained theta star')
    #     theta_star.load_state_dict(torch.load('%s/theta_star.pth' % (opt.outf)))
    # else:
    #     train_generator(theta_star, distribution=restrict_gaussian_range, threshold=opt.threshold,
    #                     iterations=50000 if not opt.debug else 1000)

# Save the weights
torch.save(theta_star.state_dict(), '%s/theta_star.pth' % (opt.outf))

print("Pretraining theta all")
# Train theta_all on all of the restricted data from the cvae
if opt.task == 'classification':
    theta_all = _classifier()
    theta_all.to(device)
    # Train theta all on the restricted set of the data
    optim_all = _optimizer(theta_all)
    sched_all = _scheduler(optim_all)
    train_classifier(theta_all, distribution=restrict_gaussian_range, iterations=10000 if not opt.debug else 100,
                     optimizer=optim_all, _scheduler=sched_all)
else:
    theta_all = VAE(device=None, architecture='fully_connected',color_channels=nc) #, activ=F.leaky_relu)
    theta_all.to(device)
    theta_all.device = device
    optim_all = _optimizer(theta_all)
    # if opt.no_train_all:
    #     print('Load pre-trained theta all')
    #     theta_all.load_state_dict(torch.load('%s/theta_all.pth' % (opt.outf)))
    # else:
    #     train_generator(theta_all, distribution=restrict_gaussian_range, threshold=opt.threshold, iterations=50000 if not opt.debug else 1000)

# Save the weights
if not opt.no_train_all:
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

print("# Diff study")

epsilons_to_test = [0.0]
disagreement_error_star, disagreement_error_star_std = [], []
disagreement_error_all, disagreement_error_all_std = [], []


def plot_valid_scores():
    # Save figure for validation scores
    plt.close()
    plt.figure()
    plt.title('Twin 1 on splits 1 and 2 for eps std' + str(eps))
    plt.xlabel('Iterations')
    plt.ylabel('Mean squared error')

    plt.plot(np.mean(pad_all_sublists(valid_scores_twin_1_on_split_1_star), 0), label='Twin 1 Star on split 1', c='b')
    plt.fill_between(range(len(np.mean(valid_scores_twin_1_on_split_1_star, 0))),
                     np.mean(valid_scores_twin_1_on_split_1_star, 0) - np.std(valid_scores_twin_1_on_split_1_star, 0),
                     np.mean(valid_scores_twin_1_on_split_1_star, 0) + np.std(valid_scores_twin_1_on_split_1_star, 0),
                     alpha=0.3, facecolor='b')
    plt.plot(np.mean(pad_all_sublists(valid_scores_twin_1_on_split_2_star), 0), label='Twin 1 Star on split 2', c='b', ls='--')
    plt.fill_between(range(len(np.mean(valid_scores_twin_1_on_split_2_star, 0))),
                     np.mean(valid_scores_twin_1_on_split_2_star, 0) - np.std(valid_scores_twin_1_on_split_2_star, 0),
                     np.mean(valid_scores_twin_1_on_split_2_star, 0) + np.std(valid_scores_twin_1_on_split_2_star, 0),
                     alpha=0.3, facecolor='b')
    plt.plot(np.mean(pad_all_sublists(valid_scores_twin_1_on_full_star), 0), label='Twin 1 Star on full', c='b',
             ls=':')
    plt.fill_between(range(len(np.mean(valid_scores_twin_1_on_full_star, 0))),
                     np.mean(valid_scores_twin_1_on_full_star, 0) - np.std(valid_scores_twin_1_on_full_star, 0),
                     np.mean(valid_scores_twin_1_on_full_star, 0) + np.std(valid_scores_twin_1_on_full_star, 0),
                     alpha=0.3, facecolor='b')
    plt.plot(np.mean(pad_all_sublists(valid_scores_twin_1_on_split_1_all), 0), label='Twin 1 All on split 1', c='r')
    plt.fill_between(range(len(np.mean(valid_scores_twin_1_on_split_1_all, 0))),
                     np.mean(valid_scores_twin_1_on_split_1_all, 0) - np.std(valid_scores_twin_1_on_split_1_all, 0),
                     np.mean(valid_scores_twin_1_on_split_1_all, 0) + np.std(valid_scores_twin_1_on_split_1_all, 0),
                     alpha=0.3, facecolor='r')
    plt.plot(np.mean(pad_all_sublists(valid_scores_twin_1_on_split_2_all), 0), label='Twin 1 All on split 2', c='r', ls='--')
    plt.fill_between(range(len(np.mean(valid_scores_twin_1_on_split_2_all, 0))),
                     np.mean(valid_scores_twin_1_on_split_2_all, 0) - np.std(valid_scores_twin_1_on_split_2_all, 0),
                     np.mean(valid_scores_twin_1_on_split_2_all, 0) + np.std(valid_scores_twin_1_on_split_2_all, 0),
                     alpha=0.3, facecolor='r')
    plt.plot(np.mean(pad_all_sublists(valid_scores_twin_1_on_full_all), 0), label='Twin 1 All on full', c='r', ls=':')
    plt.fill_between(range(len(np.mean(valid_scores_twin_1_on_full_all, 0))),
                     np.mean(valid_scores_twin_1_on_full_all, 0) - np.std(valid_scores_twin_1_on_full_all, 0),
                     np.mean(valid_scores_twin_1_on_full_all, 0) + np.std(valid_scores_twin_1_on_full_all, 0),
                     alpha=0.3, facecolor='r')
    plt.legend()
    plt.savefig(opt.outf + 'twins_1_eps_' + str(eps) + '.png')


for i, eps in enumerate(epsilons_to_test):
    disagreement_error_star_local = []
    disagreement_error_all_local = []
    valid_scores_twin_1_on_split_1_star = []
    valid_scores_twin_1_on_split_1_all = []
    valid_scores_twin_1_on_split_2_star = []
    valid_scores_twin_1_on_split_2_all = []
    valid_scores_twin_1_on_full_star = []
    valid_scores_twin_1_on_full_all  = []
    repetitions = opt.repetitions if not opt.debug else 2
    for rep in range(repetitions):
        print("Eps std", eps, 'Repetition:', rep + 1, '/', repetitions)

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
        disagreement, valid_split_1, valid_split_2, valid_split_full = train_twins(theta_star_eps, first_exp=(rep==0),
                                                                 iterations=5 if not opt.debug else 5,
                                                                 tensorboard_id='eps_std{:.4}/Star'.format(eps))
        disagreement_error_star_local.append(disagreement)
        valid_scores_twin_1_on_split_1_star.append(valid_split_1)
        valid_scores_twin_1_on_split_2_star.append(valid_split_2)
        valid_scores_twin_1_on_full_star.append(valid_split_full)

        print("Train all twins")
        disagreement, valid_split_1, valid_split_2, valid_split_full = train_twins(theta_all_eps,  first_exp=(rep==0),
                                                                 iterations=5 if not opt.debug else 5,
                                                                 tensorboard_id='eps_std{:.4}/All'.format(eps))
        # Stack results for future averaging
        disagreement_error_all_local.append(disagreement)
        valid_scores_twin_1_on_split_1_all.append(valid_split_1)
        valid_scores_twin_1_on_split_2_all.append(valid_split_2)
        valid_scores_twin_1_on_full_all.append(valid_split_full)

        plot_valid_scores()

    disagreement_error_star.append(np.mean(pad_all_sublists(disagreement_error_star_local), 0))
    disagreement_error_all.append(np.mean(pad_all_sublists(disagreement_error_all_local), 0))
    disagreement_error_star_std.append(np.std(pad_all_sublists(disagreement_error_star_local), 0))
    disagreement_error_all_std.append(np.std(pad_all_sublists(disagreement_error_all_local), 0))

    star = np.array(disagreement_error_star)
    all = np.array(disagreement_error_all)
    star_std = np.array(disagreement_error_star_std)
    all_std = np.array(disagreement_error_all_std)

    # Save figure for disagreement
    plt.close()
    plt.title('Twins disagreement for eps std' + str(eps))
    plt.xlabel('Iterations')
    plt.ylabel('Disagreement loss')
    plt.plot(star[i].T, c='b', label='Star')
    plt.fill_between(range(len(star[i].T)), star[i].T - star_std[i].T, star[i].T + star_std[i].T,
                    alpha=0.3, facecolor='b')
    plt.plot(all[i].T, c='r', label='All')
    plt.fill_between(range(len(all[i].T)), all[i].T - all_std[i].T, all[i].T + all_std[i].T,
                     alpha=0.3, facecolor='r')
    plt.legend()
    plt.savefig(opt.outf + 'eps_std{:.4}'.format(epsilons_to_test[i]) + '.png')


writer.close()
