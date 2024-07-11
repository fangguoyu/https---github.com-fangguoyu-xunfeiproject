import os
import PIL
import torch
import cv2 as cv
import numpy as np
import torchvision
from torch.utils.data import DataLoader
mninst = torchvision.datasets.MNIST(root='./minist_data',download=True,transform=torchvision.transforms.ToTensor())
data_loader = DataLoader(dataset=mninst,batch_size=8)
output_dir = './output'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
for imges,itemes in data_loader:
    for i in range(8):
        image = (imges[i,0].detach().numpy()*256).astype(np.uint8)
        label  = itemes[i].detach().item()
        cv.imwrite(os.path.join(output_dir, f"{label}_{i}.png"), image)

