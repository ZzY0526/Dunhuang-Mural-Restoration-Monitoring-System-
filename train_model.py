import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os

class SegDataset(Dataset):
    def __init__(self, img_dir, label_dir, transform=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.files = sorted(os.listdir(img_dir))
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.files[idx])
        label_path = os.path.join(self.label_dir, self.files[idx])
        image = Image.open(img_path).convert("RGB")
        label = Image.open(label_path).convert("L")

        if self.transform:
            image = self.transform(image)
            label = self.transform(label)

        return image, label

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

# 设置训练参数
transform = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor()
])

train_dataset = SegDataset('C:/Users/tv2fp3/Documents/Dunhuang_Restoration/data/Mural_seg/train/images', 'C:/Users/tv2fp3/Documents/Dunhuang_Restoration/data/Mural_seg/train/labels', transform)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

model = SimpleUNet()
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# 训练主循环
for epoch in range(10):
    for img, label in train_loader:
        output = model(img)
        label = label.float() 
        loss = criterion(output, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

torch.save(model.state_dict(), 'mural_seg_model.pth')