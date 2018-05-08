import argparse
import os
import numpy as np
import random

import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import models

def build_dataloader(batch_size):  # mix of 8 Gaussians (https://github.com/igul222/improved_wgan_training)
    scale = 2.
    centers = [
        (1, 0),
        (-1, 0),
        (0, 1),
        (0, -1),
        (1. / np.sqrt(2), 1. / np.sqrt(2)),
        (1. / np.sqrt(2), -1. / np.sqrt(2)),
        (-1. / np.sqrt(2), 1. / np.sqrt(2)),
        (-1. / np.sqrt(2), -1. / np.sqrt(2))
    ]
    centers = [(scale * x, scale * y) for x, y in centers]
    while True:
        dataset = []
        for i in xrange(batch_size):
            point = np.random.randn(2) * .02
            center = random.choice(centers)
            point[0] += center[0]
            point[1] += center[1]
            dataset.append(point)
        dataset = np.array(dataset, dtype='float32')
        dataset /= 1.414  # stdev
        yield dataset

class Logger(object):
    def __init__(self, netG, outf, nfreq=500):
        self.netG = netG
        self.outf = outf
        self.nfreq = nfreq
        self.loss = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def plot(self, i):
        fig, ax = plt.subplots()
        ax.set_xlim(-3,3)
        ax.set_ylim(-3,3)

        # --- distribution of G
        z = torch.randn(512, 2).to(self.device)
        with torch.no_grad():
            x = netG(z).detach().cpu().numpy().squeeze()
        ax.scatter(x[:,0], x[:,1], alpha=0.1)

        # --- contour of D
        x1 = x2 = np.linspace(-3, 3, 128)
        x1, x2 = np.meshgrid(x1, x2)
        x = np.hstack((x1.reshape(-1,1), x2.reshape(-1,1)))
        x = torch.tensor(x, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            y = netD(x).detach().cpu().numpy().squeeze()
        ax.contour(x1, x2, y.reshape(x1.shape))

        fig.savefig('{}/x_{}.png'.format(self.outf, i))
        plt.close(fig)

        # --- loss
        fig, ax = plt.subplots()
        ax.set_ylabel('IPM estimate')
        ax.set_xlabel('iteration')
        ax.semilogy(self.loss)
        fig.savefig('{}/loss.png'.format(self.outf))
        plt.close(fig)

    def dump(self, i, loss):
        self.loss.append(loss)
        if i % self.nfreq == 0:
            self.plot(i)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--outf', default='tmp/8gauss', help='where to save results')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--niter', type=int, default=5000)
    parser.add_argument('--niterD', type=int, default=5, help='no. updates of D per update of G')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--alpha', type=float, default=0.1, help='Lagrange multiplier')
    parser.add_argument('--rho', type=float, default=1e-5, help='quadratic weight penalty')
    args = parser.parse_args()

    cudnn.benchmark = True

    os.system('mkdir -p {}'.format(args.outf))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataloader = build_dataloader(args.batch_size)

    netG = models.FC_G().to(device)
    netD = models.FC_D().to(device)

    z = torch.FloatTensor(args.batch_size, 2).to(device)
    alpha = torch.tensor(args.alpha).to(device)
    alpha.requires_grad_()      # we will minimize this
    mone = torch.tensor(-1.0).to(device)

    optimizerG = optim.Adam(netG.parameters(), lr=args.lr, betas=(0.5, 0.9), amsgrad=True)
    optimizerD = optim.Adam(netD.parameters(), lr=args.lr, betas=(0.5, 0.9), amsgrad=True)

    logger = Logger(netG, args.outf)

    for i in range(args.niter):

        # --- train D
        for _ in range(args.niterD):
            optimizerD.zero_grad()
            data = next(dataloader)
            x_real = torch.tensor(data).to(device)
            x_fake = netG(z.normal_(0,1))
            x_real.requires_grad_()  # to compute gradD_real
            x_fake.requires_grad_()  # to compute gradD_fake

            y_real = netD(x_real)
            y_fake = netD(x_fake)
            lossE = y_real.mean() - y_fake.mean()

            # grad() does not broadcast so we compute for the sum, effect is the same
            gradD_real = torch.autograd.grad(y_real.sum(), x_real, create_graph=True)[0]
            gradD_fake = torch.autograd.grad(y_fake.sum(), x_fake, create_graph=True)[0]
            omega = 0.5*(gradD_real.view(gradD_real.size(0), -1).pow(2).sum(dim=1).mean() +
                         gradD_fake.view(gradD_fake.size(0), -1).pow(2).sum(dim=1).mean())

            loss = lossE + alpha * (1. - omega) - 0.5 * args.rho * (omega - 1.).pow(2)
            loss.backward(mone)
            optimizerD.step()
            with torch.no_grad():
                # minimize manually, note we feed 'mone' in backward()
                alpha += args.rho * alpha.grad
                alpha.grad.zero_()

        # --- train G
        optimizerG.zero_grad()
        x_fake = netG(z.normal_(0,1))
        y_fake = netD(x_fake)
        loss = -y_fake.mean()
        loss.backward()
        optimizerG.step()

        logger.dump((i+1), lossE.item())

        if (i+1) % 100 == 0:
            print "[{}/{}] loss: {:.3f}, alpha: {:.3f}, omega: {:.3f}".format((i+1), args.niter, lossE.item(), alpha.item(), omega.item())
