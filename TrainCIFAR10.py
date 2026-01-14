import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from UNet import UNet
from TrainNetwork import trainNetwork

def trainCIFAR10():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    channels = [64, 128, 256, 256]
    attention = [False, False, True, True]

    network = UNet(3, channels, attention, 2, 2, 64)
    network = network.to(device)

    transform = transforms.Compose([
        transforms.ToTensor(),                     # [0,1]
        transforms.Lambda(lambda x: x * 2.0 - 1.0) # [-1,1]
    ])

    cifar10 = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)

    loader = DataLoader(
        cifar10,
        batch_size=100,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        drop_last=True
    )

    lossF = F.mse_loss

    optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)

    network = trainNetwork(network, loader, lossF, optimizer, 100, 1, device)

if __name__ == "__main__":
    trainCIFAR10()

