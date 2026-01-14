import torch
from NoiseSchedule import cosineSchedule

def trainNetwork(network, dataLoader, lossF, optimizer, T=1000, trainSteps=10000, device="cuda"):
    sqrtA_, sqrt1mA_, *_= cosineSchedule(T, device=device)
    network.train()

    for step in range(trainSteps):
        for images, _ in dataLoader:

            images = images.to(device)
            batchSize = images.size(0)

            t = torch.randint(0, T, (batchSize,), device=device)

            eps = torch.randn_like(images)

            a = sqrtA_[t].view(batchSize, 1, 1, 1)
            b = sqrt1mA_[t].view(batchSize, 1, 1, 1)

            noisyImages = a * images + b * eps

            epsHat = network(noisyImages, t)

            lossV = lossF(epsHat, eps)

            optimizer.zero_grad(set_to_none=True)
            lossV.backward()
            optimizer.step()

        if step % 100 == 0:
            print(f"step {step}: loss {lossV.item():.4f}")
        
    return network
