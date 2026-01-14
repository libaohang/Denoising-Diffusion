import os
import torch

def saveModel(network, epoch, optimizer, scaler, lossV):
    save_dir = "./checkpoints"
    os.makedirs(save_dir, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state": network.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "loss": lossV.item(),
            "scaler": scaler.state_dict(),
        },
        f"{save_dir}/ckpt_epoch_{epoch}.pt"
    )

def loadModel(epoch, network, optimizer, device):
    ckpt = torch.load(f"checkpoints/ckpt_epoch_{epoch}.pt", map_location=device)

    network.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])

    start_epoch = ckpt["epoch"] + 1

    return start_epoch, optimizer, network