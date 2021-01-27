import torch
import torch.nn as nn


class VAETrainer:

    def __init__(self, model, loss_fct, optimizer):
        self.model = model
        self.loss_fct = loss_fct
        self.optimizer = optimizer
