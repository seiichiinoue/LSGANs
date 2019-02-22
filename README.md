# Least Squares Generative Adversarial Networks

## Description

- I used MNIST as train data.

- Mainly, learning process is same as Reguler GAN. But LSGAN, uses Least Squares criterion as below

$ min_D V_{LSGAN}(D) = \frac{1}{2} E[(D(x) - b)^2] + \frac{1}{2} E[(D(G(z)) - a)^2]$

$ min_G V_{LSGAN}(G) = \frac{1}{2} E[(D(G(z)) - c)^2]$

- In the Paper, the recommended values of $a, b, c = -1, 1, 0$ or $0, 1, 1$ 

## result

- Generated images after trained LSGANs model 30 epochs

![](./data/generated/epoch_030.png)





