from torch import nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, inChannels, outChannels=None, timeEmbDim=None):
        super().__init__()
        if outChannels == None or outChannels == inChannels:
            outChannels = inChannels
            self.residual = None
        else:
            self.residual = nn.Conv2d(inChannels, outChannels, 1)

        if timeEmbDim == None:
            self.timeEmbProj = None
        else:
            self.timeEmbProj = nn.Linear(timeEmbDim, outChannels)


        self.norm1 = nn.GroupNorm(32, inChannels)
        self.conv1 = nn.Conv2d(inChannels, outChannels, 3, padding=1)
        self.norm2 = nn.GroupNorm(32, outChannels)
        self.conv2 = nn.Conv2d(outChannels, outChannels, 3, padding=1)
        

    def forward(self, input, timeEmb=None):
        x = self.norm1(input)
        x = self.conv1(F.silu(x))

        if(timeEmb is not None):
            assert self.timeEmbProj is not None
            # temb: (batch, timeEmbDim) -> (batch, outChannels, 1, 1)
            x = x + self.timeEmbProj(F.silu(timeEmb))[:, :, None, None]

        x = self.norm2(x)
        x = self.conv2(F.silu(x))

        if(self.residual == None):
            return input + x
        else:
            return self.residual(input) + x