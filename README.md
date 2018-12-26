# Pytorch implementation of Sobolev GAN ([arXiv](https://arxiv.org/abs/1711.04894))

*Requires PyTorch 0.4+*

### Toy problem: Mixture of 8 Gaussians
`python gan_toy.py [--options]`

G and D are fully connected layers

![8 Gaussians](https://i.imgur.com/3RtQ8kn.gif)

### CIFAR10
`python gan_cifar10.py [--options]`

G is a Resnet like the one in WGAN-GP paper [https://arxiv.org/abs/1704.00028](https://arxiv.org/abs/1704.00028)

Generated samples (300+ epochs)

![CIFAR10 generated0](https://i.imgur.com/g2gUziB.png)
![CIFAR10 generated1](https://i.imgur.com/Fi8VAnU.png)

![CIFAR10 generated2](https://i.imgur.com/OGVrCSL.png)
![CIFAR10 generated3](https://i.imgur.com/0o1ak7s.png)
