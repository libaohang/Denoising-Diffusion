import torch
import math

def precompute(betas, alphas, alphaBars, eps=1e-20):
   sqrtA_ = torch.sqrt(alphaBars)
   sqrt1mA_ = torch.sqrt((1.0 - alphaBars).clamp(min=0.0))
   sqrtAInv = torch.sqrt(1.0 / alphas.clamp(min=eps))
   epsCoeff = betas / sqrt1mA_.clamp(min=eps)
   sqrtVariance = torch.sqrt(betas)
   return sqrtA_, sqrt1mA_, sqrtAInv, epsCoeff, sqrtVariance

def linearSchedule(T=1000, beta1 = 1e-4, betaT = 2e-2, device="cuda"):
    betas = torch.linspace(beta1, betaT, T, device=device)
    alphas = 1.0 - betas
    alphaBars = torch.cumprod(alphas, dim=0)
    return precompute(betas, alphas, alphaBars)

def cosineSchedule(T=1000, s=0.008, device="cuda"):
    t = torch.linspace(0, 1, T + 1, device=device, dtype=torch.float64)
    alphaBars = torch.cos(((t + s) / (1 + s)) * math.pi / 2) ** 2
    alphaBars = (alphaBars / alphaBars[0])[:-1]

    alphas = torch.ones(T, device=device, dtype=torch.float64)
    alphas[1:] = alphaBars[1:] / alphaBars[:-1]  # aBar[t] / aBar[t-1]

    betas = 1.0 - alphas
    betas = betas.clamp(1e-8, 0.999)

    alphaBars = alphaBars.to(torch.float32)
    betas = betas.to(torch.float32)
    alphas = alphas.to(torch.float32)

    return precompute(betas, alphas, alphaBars)

