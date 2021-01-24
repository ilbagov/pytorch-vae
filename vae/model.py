import torch.nn as nn
import numpy as np

H_IN = 512
W_IN = H_IN

IN_CHANNELS = 3
INIT_OUT_CHANNELS = 2**5
CONV_KERNEL_SIZE = 3
CONV_PADDING = 1

POOL_KERNEL_SIZE = 2
POOL_STRIDE = 2


class Encoder(nn.Module):

    def __init__(self, num_hidden_layers, dim_z):
        super(Encoder, self).__init__()

        hidden_dims = INIT_OUT_CHANNELS

        layers = nn.ModuleList([nn.Conv2d(in_channels=IN_CHANNELS, out_channels=hidden_dims,
                                          kernel_size=CONV_KERNEL_SIZE, padding=CONV_PADDING),
                                nn.MaxPool2d(kernel_size=POOL_KERNEL_SIZE, stride=POOL_STRIDE),
                                nn.BatchNorm2d(num_features=hidden_dims),
                                nn.ReLU()])

        h_out = np.floor((H_IN + 2*CONV_PADDING - (CONV_KERNEL_SIZE-1) - 1)+1)  # height reduction due to Conv2d
        h_out = np.floor((h_out - (POOL_KERNEL_SIZE-1) - 1)/POOL_STRIDE + 1)  # height reduction due to MaxPool2d

        w_out = np.floor((W_IN + 2*CONV_PADDING - (CONV_KERNEL_SIZE-1) - 1)+1)  # width reduction due to Conv2d
        w_out = np.floor((w_out - (POOL_KERNEL_SIZE-1) - 1)/POOL_STRIDE + 1)  # width reduction due to MaxPool2d

        for i in range(num_hidden_layers-1):
            n_in = hidden_dims
            hidden_dims *= 2

            layers.extend([nn.Conv2d(in_channels=n_in, out_channels=hidden_dims,
                                     kernel_size=CONV_KERNEL_SIZE, padding=CONV_PADDING),
                           nn.MaxPool2d(kernel_size=POOL_KERNEL_SIZE, stride=POOL_STRIDE),
                           nn.BatchNorm2d(num_features=hidden_dims),
                           nn.ReLU()])

            h_out = np.floor((h_out + 2 * CONV_PADDING - (CONV_KERNEL_SIZE - 1) - 1) + 1)  # height reduction due to Conv2d
            h_out = np.floor((h_out - (POOL_KERNEL_SIZE - 1) - 1) / POOL_STRIDE + 1)  # height reduction due to MaxPool2d

            w_out = np.floor((w_out + 2 * CONV_PADDING - (CONV_KERNEL_SIZE - 1) - 1) + 1)  # width reduction due to Conv2d
            w_out = np.floor((w_out - (POOL_KERNEL_SIZE - 1) - 1) / POOL_STRIDE + 1)  # width reduction due to MaxPool2d

        self.encoder = nn.Sequential(*layers, nn.Flatten())
        self.mu_layer = nn.Linear(int(h_out)*int(w_out)*hidden_dims, dim_z)
        self.var_layer = nn.Linear(int(h_out)*int(w_out)*hidden_dims, dim_z)

    def forward(self, x):
        x = self.encoder(x)
        mu = self.mu_layer(x)
        var = self.var_layer(x)

        return mu, var


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
