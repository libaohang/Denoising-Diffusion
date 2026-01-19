# Diffusion Model from Scratch Using PyTorch

## Description
**A diffusion probabilistic model capable of image generation**
<br>
I built a scalable diffusion model using the PyTorch library and trained and generated images from CelebA and CIFAR-10 datasets.
<br>

## Model Structure

**UNet (inChannel, channels, attention, numRes, neckLength, timeEmbDim=64)** <br>
<pre>
 Conv2d                        Output head          Width of blocks:     Has attention: <br>
    ↓                               ↑ <br>
DownBlock---skip connections---> UpBlock            channels[0]          attention[0] <br>
    ↓                               ↑ <br>
    ⋮  ᅟᅟᅟᅟ       ᅟᅟᅟᅟ        ⋮                    ⋮                     ⋮ <br>
DownBlock---skip connections---> UpBlock            channels[-2]         attention[-2] <br>
    ↓                               ↑ <br>
DownBlock → ResidualBlock → …  → UpBlock            channels[-1]         attention[-1] <br>
</pre>
Conv2d at the beginning maps *inChannel* to *channels[0]*; *channels* is a list of integers, each corresponding to the width of DownBlock and UpBlock at each level. 
*attention* is a list of boolean values, each denoting whether there is self-attention at each level.
The number of ResidualBlock connecting the bottommost DownBlock and UpBlock is determined by *neckLength*. 
Sinusoidal time embedding is injected into every DownBlock, UpBlock, and ResidualBlock. <br>

**ResidualBlock (inChannels, outChannels, timeEmbDim)** <br>
Two 3×3 convolutions with GroupNorm and SiLU activations, with a residual skip connection and additive time-embedding projection. <br>

**DownBlock (inChannels, blockChannels, numRes, timeEmbDim, attention=False, noDownsample=False)** <br>
A stack of *numRes* residual blocks followed by a strided convolution for downsampling (no downsample for the bottommost DownBlock), 
optionally including a self-attention layer. <br>

**UpBlock (inChannels, blockChannels, numRes, timeEmbDim, attention=False, noUpsample=False)** <br>
Bilinear upsampling (no upsample for the bottommost UpBlock) followed by concatenation with skip connections and residual blocks, optionally with self-attention. <br>


