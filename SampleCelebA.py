import torch
import math
from SaveLoad import loadModel
from Sampling import ddimSample, ddpmSample
from NoiseSchedule import cosineSchedule
from torchvision.utils import save_image
from UNet import UNet
from EMA import EMA

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

channels = [64, 128, 256, 512, 512]
attention = [False, False, True, True, False]

network = UNet(3, channels, attention, 3, 2, 64)
network.to(device)

ema = EMA(network, decay=0.99995)

step = 17897

loadModel(step, network, ema, "CelebA", device=device)

T = 1000

def ddim(applyEMA=False):
    if(applyEMA):
        ema.apply_shadow(network)
    *_, alphaBars = cosineSchedule(T, device=device)
    samples = ddimSample(
        network=network,
        T=T,
        alphaBars=alphaBars,
        shape=(64, 3, 64, 64),
        device=device,
        steps=100,
        eta=0.0
    )
    if(applyEMA):
        ema.restore(network)

    # save
    samples_vis = (samples + 1) / 2
    samples_vis = samples_vis.clamp(0, 1)
    save_image(samples_vis, f"samples_{int(step/1000)}k_ddim100" + ("_ema" if (applyEMA) else "") + ".png", nrow=8)
    print(f"Saved samples_{int(step/1000)}k_ddim100" + ("_ema" if (applyEMA) else "") + ".png")

def ddpm(applyEMA=False):
    if(applyEMA):
        ema.apply_shadow(network)
    alphaBars, posterior_log_var, c1, c2 = cosineSchedule(T, device=device, full=True)
    samples = ddpmSample(
        network=network,
        T=T,
        alphaBars=alphaBars,
        posterior_log_var=posterior_log_var, 
        c1=c1, 
        c2=c2,
        device=device,
        shape=(64, 3, 64, 64)
    )
    if(applyEMA):
        ema.restore(network)

    samples_vis = (samples + 1) / 2
    samples_vis = samples_vis.clamp(0, 1)
    save_image(samples_vis, f"samples_{int(step/1000)}k_ddpm" + ("_ema" if (applyEMA) else "") + ".png", nrow=8)
    print(f"Saved samples_{int(step/1000)}k_ddpm" + ("_ema" if (applyEMA) else "") + ".png")

if __name__ == "__main__":
    ddim()