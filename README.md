# Pytorch implementation of Sobolev GAN [https://arxiv.org/abs/1711.04894](https://arxiv.org/abs/1711.04894)

*Requires PyTorch 0.4+*

### Toy problem: Mixture of 8 Gaussians
`python gan_toy.py [--options]`

G and D are fully connected layers

![8 Gaussians](https://i.imgur.com/3RtQ8kn.gif)

### CIFAR10
`python gan_cifar10.py [--options]`

G and D are resnets like the one in WGAN-GP paper [https://arxiv.org/abs/1704.00028](https://arxiv.org/abs/1704.00028)

*TODO:* add gif
