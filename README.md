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
I use a cosine noise schedule and sample a random diffusion timestep for each training image. This exposes the model to varying noise levels within each batch and allows it to learn denoising across the full diffusion process simultaneously. The model is trained to predict the injected noise at each timestep, using mean squared error (MSE) loss. Gradients from this loss are backpropagated to update the network parameters. I used a batch size of 100, so the model processes 100 images every gradient step. The model is trained unconditionally, so class labels are ignored. <br>

### Datasets
I first trained the model on the CIFAR-10 dataset to debug the network and add features, as it is smaller in dimensions. After I ensured the model trains properly and the sampling process is correct, I moved on to train on the CelebA dataset, which has smoother structures and higher perceptual quality. Below are some samples of training images for reference. <br>
**CIFAR-10:** <br>
<img width="500" height="47" alt="CIFAR10_examples" src="https://github.com/user-attachments/assets/b2c1e49a-3ff1-4356-b744-a786c6e674ce" /> <br>
**CelebA:** <br>
<img width="1000" height="94" alt="CelebA_examples" src="https://github.com/user-attachments/assets/109cb24f-9e2c-465d-bf92-fb91e170b8e9" /> <br>

### Progression
As the model trained for more steps and saw more images, the samples looked progressively less distorted and more recognizable. Below is a table detailing how samples compare after the model is trained for an increasing number of steps. All samples are taken using DDPM without EMA with the same seed for consistency in comparison. <br>
<br>
**CIFAR-10:** <br>
Steps <br>
50k:       <img width="500" height="47" alt="samples_50k_ddpm" src="https://github.com/user-attachments/assets/3bca7155-8872-4c18-99a0-a3e9dbda75a6" /> <br>
100k: <img width="500" height="47" alt="samples_100k_ddpm" src="https://github.com/user-attachments/assets/3220954b-2ac1-4e06-a15e-7a5b20d07859" /> <br>
250k: <img width="500" height="47" alt="samples_250k_ddpm" src="https://github.com/user-attachments/assets/c46e20b8-5d1a-4386-acd9-29ba452a6161" /> <br>
300k: <img width="500" height="47" alt="samples_300k_ddpm" src="https://github.com/user-attachments/assets/4ea0d029-9593-44a5-a326-b464e8172239" /> <br>
350k: <img width="500" height="47" alt="samples_350k_ddpm" src="https://github.com/user-attachments/assets/cfc531c1-c86d-4eec-bd6b-d8c9b3128856" /> <br>
<br>
**CelebA:** <br>
1k: <img width="684" height="72" alt="samples_1k_ddpm_seed30" src="https://github.com/user-attachments/assets/06990df8-0bfe-47c5-8b88-aeb7a265a075" /> <br>
20k: <img width="684" height="72" alt="samples_20k_ddpm_seed30" src="https://github.com/user-attachments/assets/a5ade5b1-ee77-4821-a621-1cde8175a3fc" /> <br>
40K: <br>
60k: <br>
80k: <br>
100k: <br>
120k: <br>
140k: <br>
160k: <br>
180k: <br>
200k: <br>
220k: <br>
<br>
It's apparent how the model is denoising the same noise into similar but progressively cleaner images as it is trained for more steps. This process is more easily observed in CelebA samples since faces have a consistent structure compared to the varying perspectives and objects of CIFAR-10 images. <br>

## Sampling

### DDPM vs DDIM
DDPM sampling follows the full learned reverse diffusion process, iteratively denoising the image across all timesteps from pure noise to a final sample. This stochastic process tends to preserve fine details and produce higher perceptual quality, but it is computationally expensive due to the large number of sampling steps. <br>

DDIM accelerates sampling by using a deterministic update rule that allows the diffusion trajectory to be traversed in far fewer steps. By skipping most intermediate timesteps, DDIM significantly reduces sampling time. However, this deterministic path can oversmooth high-frequency details, sometimes resulting in less realistic samples. <br>

Below is a comparison of samples generated from the same model checkpoint after 196k training steps using DDIM with 20 steps, DDIM with 100 steps, and DDPM. <br>
<br>

**DDIM with 20 steps:** <br>
<img width="684" height="140" alt="samples_196k_ddim20_seed30" src="https://github.com/user-attachments/assets/3cf07787-6f15-48d8-b446-5d474ead36da" /> <br>
**DDIM with 100 steps:** <br>
<img width="684" height="140" alt="samples_196k_ddim100_seed30" src="https://github.com/user-attachments/assets/b423f0dd-9045-40ce-99f8-ece1aee0bf0b" /> <br>
**DDPM:** <br>
<img width="684" height="140" alt="samples_196k_ddpm_seed30" src="https://github.com/user-attachments/assets/3fc358ee-f294-403a-89a3-bafa499c7dd4" /> <br>

After 50 to 100 steps, DDIM samples achieve a similar quality as DDPM samples, especially for sampling from models in later stages of training. Note how even when all of the above samples are taken using the same seed, DDPM and DDIM samples do not match because they map the same noise to different faces. 

### EMA

**Raw weights:** <br>
<img width="500" height="47" alt="samples_100k_ddpm_seed40" src="https://github.com/user-attachments/assets/b1c99a86-b93b-4e45-8388-092521fd843d" /> <br>
**EMA weights:** <br>
<img width="500" height="47" alt="samples_100k_ddpm_ema_seed40" src="https://github.com/user-attachments/assets/e1abb103-ae78-42cd-b63d-ad6e04ae0fc8" /> <br>




## Results
CelebA: 100 DDPM samples with EMA after 229k steps: <br>
<img width="662" height="662" alt="100CelebA_samples_229k_ddpm_ema" src="https://github.com/user-attachments/assets/45b4163b-84f3-4792-b8e0-ca39a1b32a5d" /> <br>
<br>
CIFAR-10: 100 DDPM samples with EMA after 350k steps: <br> 
<img width="500" height="500" alt="CIFAR10_samples_350k_ddpm_ema" src="https://github.com/user-attachments/assets/3207e00d-8f75-4e87-a2f5-6559e58652c4" /> <br>


## References
https://nathanbaileyw.medium.com/a-look-at-diffusion-models-79bd7e789964
https://arxiv.org/pdf/2006.11239
https://arxiv.org/abs/2010.02502
