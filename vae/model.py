import torch
import torch.nn as nn
import numpy as np
from torch.distributions.multivariate_normal import MultivariateNormal

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

        self.encoder_net = nn.Sequential(*layers, nn.Flatten())
        self.mu_layer = nn.Linear(int(h_out)*int(w_out)*hidden_dims, dim_z)
        self.var_layer = nn.Linear(int(h_out)*int(w_out)*hidden_dims, dim_z)

    def forward(self, x):
        x = self.encoder_net(x)
        mu = self.mu_layer(x)
        var = self.var_layer(x)

        return mu, var


class Decoder(nn.Module):

    def __init__(self, num_hidden_layers, dim_z):
        super(Decoder, self).__init__()

        hidden_dims = INIT_OUT_CHANNELS
        h_out = H_IN
        w_out = W_IN

        h_out = np.floor((h_out + 2 * CONV_PADDING - (CONV_KERNEL_SIZE - 1) - 1) + 1)  # height reduction due to Conv2d
        h_out = np.floor((h_out - (POOL_KERNEL_SIZE - 1) - 1) / POOL_STRIDE + 1)  # height reduction due to MaxPool2d

        w_out = np.floor((w_out + 2 * CONV_PADDING - (CONV_KERNEL_SIZE - 1) - 1) + 1)  # width reduction due to Conv2d
        w_out = np.floor((w_out - (POOL_KERNEL_SIZE - 1) - 1) / POOL_STRIDE + 1)  # width reduction due to MaxPool2d

        for i in range(num_hidden_layers-1):
            hidden_dims *= 2

            h_out = np.floor((h_out + 2 * CONV_PADDING - (CONV_KERNEL_SIZE - 1) - 1) + 1)  # height reduction due to Conv2d
            h_out = np.floor((h_out - (POOL_KERNEL_SIZE - 1) - 1) / POOL_STRIDE + 1)  # height reduction due to MaxPool2d

            w_out = np.floor((w_out + 2 * CONV_PADDING - (CONV_KERNEL_SIZE - 1) - 1) + 1)  # width reduction due to Conv2d
            w_out = np.floor((w_out - (POOL_KERNEL_SIZE - 1) - 1) / POOL_STRIDE + 1)  # width reduction due to MaxPool2d

        self.input_layer = nn.Linear(dim_z, int(h_out)*int(w_out)*hidden_dims)

        self.h_input = int(h_out)
        self.w_input = int(w_out)
        self.channels_input = int(hidden_dims)

        hidden_dims_decoder = hidden_dims
        layers = nn.ModuleList([nn.ConvTranspose2d(in_channels=int(hidden_dims_decoder),
                                                   out_channels=int(hidden_dims_decoder/2),
                                                   kernel_size=CONV_KERNEL_SIZE, padding=CONV_PADDING,
                                                   stride=2, output_padding=1),
                                nn.BatchNorm2d(num_features=int(hidden_dims_decoder/2)),
                                nn.ReLU()])

        for i in range(num_hidden_layers-1):
            hidden_dims_decoder = hidden_dims_decoder/2
            layers.extend([nn.ConvTranspose2d(in_channels=int(hidden_dims_decoder),
                                              out_channels=int(hidden_dims_decoder/2),
                                              kernel_size=CONV_KERNEL_SIZE, padding=CONV_PADDING,
                                              stride=2, output_padding=1),
                           nn.BatchNorm2d(num_features=int(hidden_dims_decoder/2)),
                           nn.ReLU()])

        # add final layer
        layers.extend([nn.Conv2d(in_channels=int(hidden_dims_decoder/2), out_channels=3,
                                 kernel_size=CONV_KERNEL_SIZE, padding=CONV_PADDING),
                       nn.Tanh()])

        self.decoder_net = nn.Sequential(*layers)

    def forward(self, x):
        x = self.input_layer(x)
        x = x.view(-1, self.channels_input, self.w_input, self.h_input)
        x = self.decoder_net(x)

        return x


class VAE(nn.Module):

    def __init__(self, num_hidden_layers, dim_z):
        super(VAE, self).__init__()
        self.encoder = Encoder(num_hidden_layers, dim_z)
        self.decoder = Decoder(num_hidden_layers, dim_z)
        self.dist = MultivariateNormal(loc=torch.ones(dim_z), covariance_matrix=torch.eye(dim_z))
        pass

    def forward(self, x):
        # TODO differentiate between training/inference
        mu, sigma = self.encoder(x)
        epsilon = self.dist.rsample()

        z = mu + sigma*epsilon
        x = self.decoder(z)

        return x

