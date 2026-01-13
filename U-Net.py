from torch import nn
from Blocks import ResidualBlock, UpBlock, DownBlock

class UNet(nn.Module):
    def __init__(self, inChannel, channels, attention, numRes, neckLength, timeEmbDim=None):
        super().__init__()
        assert len(channels) >= 3
        assert len(attention) == len(channels)

        self.stem = nn.Conv2d(inChannel, channels[0], 3, padding=1)

        self.downBlocks = nn.ModuleList(
            [DownBlock(channels[0], channels[0], numRes, timeEmbDim, attention[0])] +
            [DownBlock(channels[i-1], channels[i], numRes, timeEmbDim, attention[i]) for i in range(1, len(channels))]
        )

        self.bottleneck = nn.ModuleList(
            [ResidualBlock(channels[-1], timeEmbDim=timeEmbDim) for _ in range(neckLength)]
        )

        self.upBlocks = nn.ModuleList(
            [UpBlock(channels[i], channels[i-1], numRes, timeEmbDim, attention[i]) for i in range(len(channels)-1, 0, -1)] +
            [UpBlock(channels[0], channels[0], numRes, timeEmbDim, attention[0])]
        )

        self.head = nn.Sequential(
            nn.GroupNorm(32, channels[0]),
            nn.SiLU(),
            nn.Conv2d(channels[0], inChannel, 3, padding=1)
        )
        
    def forward(self, image, timeEmb=None):
        residuals = []
        image = self.stem(image)

        for block in self.downBlocks:
            image, residual = block(image, timeEmb)
            # residual is a list of all residuals in a block
            residuals.append(residual)
        
        for block in self.bottleneck:
            image = block(image, timeEmb)

        residuals = reversed(residuals)
        for block, residual in zip(self.upBlocks, residuals, strict=True):
            image = block(image, residual, timeEmb)

        image = self.head(image)
        return image