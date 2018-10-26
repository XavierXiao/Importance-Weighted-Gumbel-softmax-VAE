# Importance-Weighted Gumbel-softmax-VAE

This is a Pytorch implementation of IWAE [1] with categorical latent varibles parametrized by Gumbel-softmax distribution[2]. The number of  categorical latent variables is 20, and each is a 10-categorical variable. The network structure is 784-512-256-20x10-256-512-784, with ReLU activation. Batch size is 100, and ADAM optimizer is used with learning rate schedule proposed by [1]. I only test k=5, which means 5 samples from posterior are drawn each time.

The model is trained 50 epoches, and being tested on MNIST test set. The NLL is computed as [1]. The importance weight slightly improve the NLL compared to VAE with multiple samples.

| Gumbel_VAE | Gumbel_IWAE |
| ------------- | ------------- |
| 92.88  | 92.31  |

The IWAE code is partly based on implementation of Xinqiang Ding: https://github.com/xqding/Importance_Weighted_Autoencoders
Reference:

[1]Burda, Yuri, Roger Grosse, and Ruslan Salakhutdinov. "Importance weighted autoencoders." arXiv preprint arXiv:1509.00519 (2015).

[2]Jang, Eric, Shixiang Gu, and Ben Poole. "Categorical reparameterization with gumbel-softmax." arXiv preprint arXiv:1611.01144 (2016).
