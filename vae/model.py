from torch import nn


class Encoder(nn.Module):

    def __init__(self):
        super(Encoder, self).__init__()
        pass

    def forward(self, x):
        pass


class Decoder(nn.Module):

    def __init__(self):
        super(Decoder, self).__init__()
        pass

    def forward(self, x):
        pass


class VAE(nn.Module):

    def __init__(self):
        super(VAE, self).__init__()
        pass

    def forward(self, x):
        mu, sigma = self.encoder(x)
        epsilon = 0 # TODO: implement sampling

        z = mu + sigma*epsilon
        x = self.decoder(z)

        return x

    def forward_trained(self, z):
        x = self.decoder(z)

        return z
