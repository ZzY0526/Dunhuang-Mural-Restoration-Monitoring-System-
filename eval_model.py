import torch
from train_model import SimpleUNet, SegDataset, transform
from PIL import Image
import matplotlib.pyplot as plt

test_dataset = SegDataset('C:/Users/tv2fp3/Documents/Dunhuang_Restoration/data/Mural_seg/test/images', 'C:/Users/tv2fp3/Documents/Dunhuang_Restoration/data/Mural_seg/test/labels', transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1)

model = SimpleUNet()
model.load_state_dict(torch.load('mural_seg_model.pth'))
model.eval()

with torch.no_grad():
    for img, label in test_loader:
        output = model(img)
        pred_mask = torch.sigmoid(output)[0][0].numpy()
        gt_mask = label[0][0].numpy()

        fig, axs = plt.subplots(1, 3)
        axs[0].imshow(img[0].permute(1, 2, 0))
        axs[1].imshow(gt_mask, cmap='gray')
        axs[2].imshow(pred_mask > 0.5, cmap='gray')
        for ax in axs: ax.axis('off')
        plt.show()
        break