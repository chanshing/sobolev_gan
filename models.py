import torch.nn as nn

class FC_G(nn.Module):
    def __init__(self, idim=2, odim=2, hidden_dim=512):
        super(FC_G, self).__init__()

        main = nn.Sequential(
            nn.Linear(idim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, odim),
        )
        self.main = main

    def forward(self, noise):
        output = self.main(noise)
        return output

class FC_D(nn.Module):
    def __init__(self, idim=2, hidden_dim=512):
        super(FC_D, self).__init__()

        main = nn.Sequential(
            nn.Linear(idim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, 1),
        )
        self.main = main

    def forward(self, input):
        output = self.main(input)
        return output.view(-1)

class Resnet_G(nn.Module):      # https://github.com/caogang/wgan-gp/blob/master/gan_cifar10.py
    def __init__(self, hidden_dim=128):
        super(Resnet_G, self).__init__()
        self.hidden_dim = hidden_dim

        preprocess = nn.Sequential(
            nn.Linear(128, 4 * 4 * 4 * hidden_dim),
            nn.BatchNorm1d(4 * 4 * 4 * hidden_dim),
            nn.ReLU(True),
        )

        block1 = nn.Sequential(
            nn.ConvTranspose2d(4 * hidden_dim, 2 * hidden_dim, 2, stride=2),
            nn.BatchNorm2d(2 * hidden_dim),
            nn.ReLU(True),
        )
        block2 = nn.Sequential(
            nn.ConvTranspose2d(2 * hidden_dim, hidden_dim, 2, stride=2),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(True),
        )
        deconv_out = nn.ConvTranspose2d(hidden_dim, 3, 2, stride=2)

        self.preprocess = preprocess
        self.block1 = block1
        self.block2 = block2
        self.deconv_out = deconv_out
        self.tanh = nn.Tanh()

    def forward(self, input):
        output = self.preprocess(input)
        output = output.view(-1, 4 * self.hidden_dim, 4, 4)
        output = self.block1(output)
        output = self.block2(output)
        output = self.deconv_out(output)
        output = self.tanh(output)
        return output.view(-1, 3, 32, 32)


class Convnet_D(nn.Module):      # https://github.com/caogang/wgan-gp/blob/master/gan_cifar10.py
    def __init__(self, hidden_dim=128):
        super(Convnet_D, self).__init__()
        self.hidden_dim = hidden_dim

        main = nn.Sequential(
            nn.Conv2d(3, hidden_dim, 3, 2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dim, 2 * hidden_dim, 3, 2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(2 * hidden_dim, 4 * hidden_dim, 3, 2, padding=1),
            nn.LeakyReLU(),
        )

        self.main = main
        self.linear = nn.Linear(4*4*4*hidden_dim, 1)

    def forward(self, input):
        output = self.main(input)
        output = output.view(-1, 4*4*4*self.hidden_dim)
        output = self.linear(output)
        return output
