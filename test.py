import torch


file = torch.load(r"output\checkpoints\best_model\checkpoint_epoch_39.pth", weights_only=False)

print(file)