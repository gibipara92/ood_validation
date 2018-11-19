
criterion = F.nll_loss

# setup optimizer
optimizer = optim.RMSprop(theta_star.parameters(), lr=0.1)  # , momentum=0.5) #, lr=opt.lr, betas=(opt.beta1, 0.999))
schedulerC = torch.optim.lr_scheduler.StepLR(optimizerC, step_size=150000, gamma=0.2)
# optimizerC = optim.Adam(causal.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
# optimizerAC = optim.Adam(anticausal.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerAC = optim.RMSprop(theta_all.parameters(), lr=0.1)  # , momentum=0.5) #, lr=opt.lr, betas=(opt.beta1, 0.999))
schedulerAC = torch.optim.lr_scheduler.StepLR(optimizerAC, step_size=150000, gamma=0.2)


########################################################################################################################
#                   Start training
########################################################################################################################

def plotter(iteration):
    theta_star.eval()
    theta_all.eval()

    plt.figure(figsize=(17, 5.5))
    with torch.no_grad():
        plt.subplot(1, 2, 1)
        plt.axvspan(*iid_available_range, alpha=0.15, color='green')
        plt.scatter(fixed_x_test_non_iid_numpy, fixed_y_test_non_iid_numpy, label='gt')
        plt.scatter(fixed_x_test_non_iid_numpy, theta_star(fixed_x_test_non_iid).cpu().numpy(), label='iid', c='r')
        plt.ylim(-3, 3)
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.axvspan(*non_iid_training_range, alpha=0.15, color='green')
        plt.axvspan(*validation_range, alpha=0.15, color='red')
        plt.scatter(fixed_x_test_non_iid_numpy, fixed_y_test_non_iid_numpy, label='gt')
        plt.scatter(fixed_x_test_non_iid_numpy, theta_all  (fixed_x_test_non_iid).cpu().numpy(), label='non iid')
        plt.legend()
        plt.ylim(-3, 3)
        plt.tight_layout()
        plt.savefig('%s/plot_%06d.png' % (opt.outf, iteration), bbox_inches='tight')
        plt.close()


def error_on_valid():
    with torch.no_grad():
        theta_all  _output = theta_all  (fixed_x_valid_non_iid)
        err_theta_all   = criterion(theta_all  _output, fixed_y_valid_non_iid)
        writer.add_scalar('Non_iid/valid', err_theta_all.item(), iterations)
    return err_theta_all.item()


def error_on_test():
    with torch.no_grad():
        theta_star_output = theta_star(fixed_x_test_non_iid)
        err_theta_star = criterion(theta_star_output, fixed_y_test_non_iid)
        writer.add_scalar('Iid/holdout', err_theta_star.item(), iterations)

        theta_all  _output = theta_all  (fixed_x_test_non_iid)
        err_theta_all   = criterion(theta_all  _output, fixed_y_test_non_iid)
        writer.add_scalar('Non_iid/holdout', err_theta_all.item(), iterations)
    return err_theta_star.item(), err_theta_all.item()


iterations = -1
cooldown = -1
for i in range(opt.niter):
    iterations += 1

    # schedulerC.step()
    # schedulerAC.step()
    # writer.add_scalar('stats/curr_iterations', i, iterations)

    theta_star.train()
    theta_all.train()

    # reset grads
    theta_star.zero_grad()
    theta_all.zero_grad()

    # sample from ground truth
    gt_x_iid, gt_y_iid = torch.Tensor(ground_truth.sample(n=opt.batchSize, x_range=iid_available_range)).to(device)
    gt_x_non_iid, gt_y_non_iid = torch.Tensor(ground_truth.sample(n=opt.batchSize, x_range=non_iid_training_range)).to(
        device)
    # if opt.cuda:
    #     real_cpu = real_cpu.to(device)

    ###########################
    # (1) Update Causal network
    causal_output = theta_star(gt_x_iid)
    err_causal = criterion(causal_output, gt_y_iid)
    err_causal.backward()
    optimizerC.step()

    ############################
    # (2) Update Anti-Causal network

    anticausal_output = theta_all  (gt_x_non_iid)
    err_anticausal = criterion(anticausal_output, gt_y_non_iid)
    # reflectors_loss = compute_reflector_loss(theta_all  ) * abs(theta_all.ratio - 1.0)
    # err_anticausal = reflectors_loss + err_anticausal_no_refl
    err_anticausal.backward()
    optimizerAC.step()

    theta_star.eval()
    theta_all.eval()

    # if not iterations % 1000:
    #     theta_all.apply(init_weights)

    if not iterations % 10:
        writer.add_scalar('Iid/train_loss', err_causal.item(), iterations)
        writer.add_scalar('Non_iid/train_loss', err_anticausal.item(), iterations)

    if not iterations % 100 and opt.input_size == 1 and not opt.dcgan:
        plotter(iterations)

    #################################################################
    # Save stats
    #################################################################

    if not iterations % 10 and iterations > 0:
        print('[%d/%d] Loss_C: %.8f Loss_AC: %.8f '  # D(x): %.4f D(G(z)): %.4f / %.4f'
              % (iterations, opt.niter,
                 err_causal.item(), err_anticausal.item()))
        err_causal_test, err_anticausal_test = error_on_test()
        err_anticausal_valid = error_on_valid()

# # do checkpointing
# torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (opt.outf, epoch))
# torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (opt.outf, epoch))