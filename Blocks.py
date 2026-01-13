import torch
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
            # temb: (1, timeEmbDim) -> (batch, outChannels, 1, 1)
            x = x + self.timeEmbProj(F.silu(timeEmb))[:, :, None, None]

        x = self.norm2(x)
        x = self.conv2(F.silu(x))

        if(self.residual == None):
            return input + x
        else:
            return self.residual(input) + x
        

class DownBlock(nn.Module):
    def __init__(self, inChannels, blockChannels, timeEmbDim, numRes):
        super().__init__()
        self.resBlocks = nn.ModuleList(
            [ResidualBlock(inChannels, blockChannels, timeEmbDim)] +
            [ResidualBlock(blockChannels, timeEmbDim=timeEmbDim) for _ in range(numRes - 1)]
        )
        self.down = nn.Conv2d(blockChannels, blockChannels, 3, stride=2, padding=1)

    def forward(self, input, timeEmb=None):
        residuals = []
        x = input
        for resBlock in self.resBlocks:
            x = resBlock(x, timeEmb)
            residuals.append(x)
        x = self.down(x)
        return x, residuals


class UpBlock(nn.Module):
    def __init__(self, inChannels, blockChannels, timeEmbDim, numRes):
        super().__init__()
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        # Residual blocks take in concatenated input
        self.resBlocks = nn.ModuleList(
            [ResidualBlock(inChannels + blockChannels, blockChannels, timeEmbDim)] +
            [ResidualBlock(blockChannels * 2, blockChannels, timeEmbDim=timeEmbDim) for _ in range(numRes - 1)]
        )
    
    def forward(self, input, residuals, timeEmb=None):
        x = input
        x = self.up(x)
        residuals = reversed(residuals)
        assert len(residuals) == len(self.resBlocks)
        for i, resBlock in enumerate(self.resBlocks):
            x = torch.cat((x, residuals[i]), dim=1)
            x = resBlock(x, timeEmb)
        return x
        