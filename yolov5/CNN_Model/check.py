import torch


state = torch.load("./train/exp1/best_model.pth", map_location='cuda:0') 
for k, v in state.items():
    print(f"{k}: {v.shape}")

