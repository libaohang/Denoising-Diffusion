import torch
from NoiseSchedule import cosineSchedule

def trainNetwork(network, images, lossF, optimizer, T=1000, trainSteps=10000, batchSize=100, device="cuda"):
    sqrtA_, sqrt1mA_, *_= cosineSchedule(T, device=device)
    network.train()
    images = images.to(device)

    numSamples = images.size(0)

    for step in range(trainSteps):
        # Random batch
        index = torch.randint(0, numSamples, (batchSize,), device=device)
        batch = images[index]

        t = torch.randint(0, T, (batchSize,), device=device)

        eps = torch.randn_like(batch)

        a = sqrtA_[t].view(-1, 1, 1, 1)
        b = sqrt1mA_[t].view(-1, 1, 1, 1)

        noisyImages = a * batch + b * eps

        epsHat = network(noisyImages, t)

        lossV = lossF(epsHat, eps)

        optimizer.zero_grad(set_to_none=True)
        lossV.backward()
        optimizer.step()

        if step % 100 == 0:
            print(f"step {step}: loss {lossV.item():.4f}")
        
    return network
