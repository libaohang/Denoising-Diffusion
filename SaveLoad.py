import os
import torch

def saveModel(network, ema, step, optimizer, scaler, lossV, all=False):
    save_dir = "./checkpoints"
    os.makedirs(save_dir, exist_ok=True)
    if(all):
        torch.save(
            {
                "step": step,
                "ema": ema.shadow,
                "model_state": network.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "loss": lossV.item(),
                "scaler": scaler.state_dict(),
            },
            f"{save_dir}/ckpt_step_{step}.pt"
        )
    else:
        torch.save(
            {
                "step": step,
                "ema": ema.shadow,
                "model_state": network.state_dict(),
                "loss": lossV.item(),
            },
            f"{save_dir}/ckpt_step_{step}.pt"
        )

def loadModel(step, network, ema, optimizer=None, scaler=None, device="cuda"):
    ckpt = torch.load(f"checkpoints/ckpt_step_{step}.pt", map_location=device)

    network.load_state_dict(ckpt["model_state"])
    ema.shadow = ckpt["ema"]
    if(optimizer is not None):
        optimizer.load_state_dict(ckpt["optimizer_state"])
    if(scaler is not None):
        scaler.load_state_dict(ckpt["scaler"])

    start_step = ckpt["step"] + 1

    return start_step, optimizer, network