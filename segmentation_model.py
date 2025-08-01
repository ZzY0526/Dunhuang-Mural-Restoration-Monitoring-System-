import torch
import torch.nn as nn
import torchvision.transforms as T
import numpy as np
from PIL import Image

class SimpleUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 2, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 1, 1)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
class UNet(nn.Module):

    def predict_damage_mask(image, model):
        transform = T.Compose([T.ToTensor()])
        input_tensor = transform(image).unsqueeze(0)
        with torch.no_grad():
            output = torch.sigmoid(model(input_tensor))
        mask = output.squeeze().numpy()
        mask_img = Image.fromarray((mask * 255).astype(np.uint8)).convert("L")
        return mask_img