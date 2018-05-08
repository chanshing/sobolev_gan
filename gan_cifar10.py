import argparse
import os
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import models


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--outf', default='tmp/cifar10', help='where to save results')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--niterD', type=int, default=5, help='no. updates of D per update of G')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--alpha', type=float, default=0.1, help='Lagrange multiplier')
    parser.add_argument('--rho', type=float, default=1e-5, help='quadratic weight penalty')
    args = parser.parse_args()

    cudnn.benchmark = True

    os.system('mkdir -p {}'.format(args.outf))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    IMAGE_SIZE = 32
    dataset = dset.CIFAR10(root='cifar10', download=True,
                           transform=transforms.Compose([
                               transforms.Resize(IMAGE_SIZE),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                             shuffle=True, num_workers=2, drop_last=True)

    # Resnets from WGAN-GP paper
    netG = models.Resnet_G().to(device)
    netD = models.Resnet_D().to(device)

    NZ = 128
    z = torch.FloatTensor(args.batch_size, NZ).to(device)
    alpha = torch.tensor(args.alpha).to(device)
    alpha.requires_grad_()      # we will minimize this
    mone = torch.tensor(-1.0).to(device)

    optimizerG = optim.Adam(netG.parameters(), lr=args.lr, betas=(0.5, 0.9), amsgrad=True)
    optimizerD = optim.Adam(netD.parameters(), lr=args.lr, betas=(0.5, 0.9), amsgrad=True)

    losses = []
    for epoch in range(args.epochs):
        for i, data in enumerate(dataloader):

            # --- train D
            for _ in range(args.niterD):
                optimizerD.zero_grad()
                x_real = data[0].to(device)
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

            # --- logging
            losses.append(lossE.item())

            if (i+1) % 100 == 0:
                print "epoch: {} | [{}/{}] loss: {:.3f}, alpha: {:.3f}, omega: {:.3f}".format(
                    epoch, (i+1), int(len(dataset)/args.batch_size), lossE.item(), alpha.item(), omega.item())

        # generated images and loss curve
        vutils.save_image(x_fake, '{}/x_{}.png'.format(args.outf, epoch), normalize=True)
        fig, ax = plt.subplots()
        ax.set_ylabel('IPM estimate')
        ax.set_xlabel('iteration')
        ax.semilogy(losses)
        fig.savefig('{}/loss.png'.format(args.outf))
        plt.close(fig)
