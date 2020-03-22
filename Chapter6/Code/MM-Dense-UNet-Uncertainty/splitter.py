import torch.nn as nn

class Splitter(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
    	return self.model(x[0], x[1])
