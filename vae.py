import  torch
from    torch import nn
from    torch.nn import functional as F

# from https://github.com/dragen1860/pytorch-mnist-vae

# start

class Encoder(nn.Module):


    def __init__(self, imgsz, n_hidden, n_output, keep_prob):
        super(Encoder, self).__init__()

        self.imgsz = imgsz
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.keep_prob = keep_prob

        self.net = nn.Sequential(
            nn.Linear(imgsz, n_hidden),
            nn.ELU(inplace=True),
            nn.Dropout(1-keep_prob),

            nn.Linear(n_hidden, n_hidden),
            nn.Tanh(),
            nn.Dropout(1-keep_prob),

            nn.Linear(n_hidden, n_output*2)

        )

    def forward(self, x):
        """

        :param x:
        :return:
        """
        mu_sigma = self.net(x)


        # The mean parameter is unconstrained
        mean = mu_sigma[:, :self.n_output]
        # The standard deviation must be positive. Parametrize with a softplus and
        # add a small epsilon for numerical stability
        stddev = 1e-6 + F.softplus(mu_sigma[:, self.n_output:])


        return mean, stddev



class Decoder(nn.Module):


    def __init__(self, dim_z, n_hidden, n_output, keep_prob):
        super(Decoder, self).__init__()

        self.dim_z = dim_z
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.keep_prob = keep_prob

        self.layer1 = nn.Sequential(
            nn.Linear(dim_z, n_hidden),
            nn.Tanh(),
            nn.Dropout(1-keep_prob)
        )

        self.layer2 = nn.Sequential(
            nn.Linear(n_hidden, n_hidden),
            nn.ELU(),
            nn.Dropout(1-keep_prob),
        )

        self.layer3 = nn.Sequential(
            nn.Linear(n_hidden, n_output),
            nn.Sigmoid()
        )

        self.net = nn.Sequential(
            *self.layer1,
            *self.layer2,
            *self.layer3
        )

        # self.net = nn.Sequential(
        #     nn.Linear(dim_z, n_hidden),
        #     nn.Tanh(),
        #     nn.Dropout(1-keep_prob),

        #     nn.Linear(n_hidden, n_hidden),
        #     nn.ELU(),
        #     nn.Dropout(1-keep_prob),

        #     nn.Linear(n_hidden, n_output),
        #     nn.Sigmoid()
        # )

    def forward(self, h):
        """

        :param h:
        :return:
        """
        return self.net(h)

    def getHidden(self, h):
        return self.layer1(h)


def init_weights(encoder, decoder):

    def init_(m):
        with torch.no_grad():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0.0)

    for m in encoder.modules():
        m.apply(init_)
    for m in decoder.modules():
        m.apply(init_)

    print('weights inited!')



def get_ae(encoder, decoder, x):
    # encoding
    mu, sigma = encoder(x)
    # sampling by re-parameterization technique
    z = mu + sigma * torch.randn_like(mu)

    # decoding
    y = decoder(z)
    y = torch.clamp(y, 1e-8, 1 - 1e-8)

    return y



def get_z(encoder, x):

    # encoding
    mu, sigma = encoder(x)
    # sampling by re-parameterization technique
    z = mu + sigma * torch.randn_like(mu)

    return z

def get_loss(encoder, decoder, x, x_target):
    """

    :param encoder:
    :param decoder:
    :param x: input
    :param x_hat: target
    :param dim_img:
    :param dim_z:
    :param n_hidden:
    :param keep_prob:
    :return:
    """
    batchsz = x.size(0)
    # encoding
    mu, sigma = encoder(x)
    # sampling by re-parameterization technique
    z = mu + sigma * torch.randn_like(mu)

    # decoding
    y = decoder(z)
    y = torch.clamp(y, 1e-8, 1 - 1e-8) # 0到1


    # loss
    # marginal_likelihood2 = torch.sum(x_target * torch.log(y) + (1 - x_target) * torch.log(1 - y)) / batchsz
    marginal_likelihood = -F.binary_cross_entropy(y, x_target, reduction='sum') / batchsz
    # print(marginal_likelihood2.item(), marginal_likelihood.item())

    KL_divergence = 0.5 * torch.sum(
                                torch.pow(mu, 2) +
                                torch.pow(sigma, 2) -
                                torch.log(1e-8 + torch.pow(sigma, 2)) - 1
                               ).sum() / batchsz

    ELBO = marginal_likelihood - KL_divergence

    loss = -ELBO

    return y, z, loss, marginal_likelihood, KL_divergence

# end

class Classifier(nn.Module):
  def __init__(self, n_hidden):
    super().__init__()
    self.dims = [n_hidden, 128]
    layers = []
    for i in range(1, len(self.dims)):
      layers += [
        # nn.BatchNorm1d(self.dims[i-1]),
        nn.Linear(self.dims[i-1], self.dims[i]),
        nn.ELU()
      ]
    layers += [
      nn.Dropout(0.01), 
      nn.Linear(self.dims[-1], 10)
    ]
    self.layers = nn.Sequential(*layers)

  def forward(self, x):
    logits = self.layers(x)
    return logits
