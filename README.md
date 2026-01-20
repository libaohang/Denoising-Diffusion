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

## Training
I use a cosine noise schedule and sample a random diffusion timestep for each training image. This exposes the model to varying noise levels within each batch and allows it to learn denoising across the full diffusion process simultaneously. The model is trained to predict the injected noise at each timestep, using mean squared error (MSE) loss. Gradients from this loss are backpropagated to update the network parameters. I used a batch size of 100, so the model processes 100 images every gradient step. <br>

### Datasets
I first trained the model on the CIFAR-10 dataset to debug the network and add features, as it is smaller in dimensions. After I ensured the model trains properly and the sampling process is correct, I moved on to train on the CelebA dataset, which has smoother structures and higher perceptual quality. Below are some samples of training images for reference. <br>
CIFAR-10: <br>
<img width="500" height="47" alt="CIFAR10_examples" src="https://github.com/user-attachments/assets/b2c1e49a-3ff1-4356-b744-a786c6e674ce" /> <br>
CelebA: <br>
<img width="1000" height="94" alt="CelebA_examples" src="https://github.com/user-attachments/assets/109cb24f-9e2c-465d-bf92-fb91e170b8e9" /> <br>

### Progression
As the model trained for more steps and saw more images, the samples looked progressively less distorted and recognizable. Below is a table detailing how samples compare after the model is trained for an increasing number of steps. All samples are taken using DDPM without EMA for consistency. <br>
**CIFAR-10:** <br>
Steps <br>
50k:       <img width="500" height="47" alt="samples_50k_ddpm" src="https://github.com/user-attachments/assets/3bca7155-8872-4c18-99a0-a3e9dbda75a6" /> <br>
100k: <img width="500" height="47" alt="samples_100k_ddpm" src="https://github.com/user-attachments/assets/3220954b-2ac1-4e06-a15e-7a5b20d07859" /> <br>
250k: <img width="500" height="47" alt="samples_250k_ddpm" src="https://github.com/user-attachments/assets/c46e20b8-5d1a-4386-acd9-29ba452a6161" /> <br>
300k: <img width="500" height="47" alt="samples_300k_ddpm" src="https://github.com/user-attachments/assets/4ea0d029-9593-44a5-a326-b464e8172239" /> <br>
350k: <img width="500" height="47" alt="samples_350k_ddpm" src="https://github.com/user-attachments/assets/cfc531c1-c86d-4eec-bd6b-d8c9b3128856" /> <br>




## References
https://nathanbaileyw.medium.com/a-look-at-diffusion-models-79bd7e789964
https://arxiv.org/pdf/2006.11239
https://arxiv.org/abs/2010.02502
