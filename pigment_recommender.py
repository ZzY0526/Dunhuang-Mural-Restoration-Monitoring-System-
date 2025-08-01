from sklearn.cluster import KMeans
import numpy as np
from PIL import Image

def get_dominant_color(image, mask):
    # 转换为 numpy 数组
    image_np = np.array(image)
    mask_np = np.array(mask)

    # 修正尺寸不匹配
    if mask_np.shape[:2] != image_np.shape[:2]:
        mask_img = Image.fromarray(mask_np)
        mask_img = mask_img.resize((image_np.shape[1], image_np.shape[0]))
        mask_np = np.array(mask_img)

    # 保证掩膜是灰度
    if mask_np.ndim == 3:
        mask_np = mask_np[:, :, 0]

    # 生成布尔掩码
    damage_mask = mask_np > 100

    # 提取损伤区域 RGB 像素
    if image_np.ndim == 3 and image_np.shape[2] == 3:
        r = image_np[:, :, 0][damage_mask]
        g = image_np[:, :, 1][damage_mask]
        b = image_np[:, :, 2][damage_mask]
        pixels = np.stack((r, g, b), axis=-1)
    else:
        p = image_np[damage_mask]
        pixels = np.stack((p, p, p), axis=-1)

    # 没有有效像素时返回默认值
    if pixels.shape[0] == 0:
        return (128, 128, 128)

    # 聚类提取主色
    kmeans = KMeans(n_clusters=1, n_init=10).fit(pixels)
    dom_color = kmeans.cluster_centers_[0]
    return tuple(map(int, dom_color))

def recommend_pigment(color):
    r, g, b = color

    if r < 60 and g < 60 and b < 60:
        return "Recommended to use Carbon Black or Carbon Black, suitable for line drawing or shadow rendering."
    elif r > 220 and g > 220 and b > 220:
        return "Advised to use Muscovite or chalk-based pigments, suitable for highlights or bright background areas."
    elif abs(r - g) < 15 and abs(g - b) < 15 and r < 180:
        return "Recommended to use Graphite Gray, ideal for backgrounds or restoration masking layers."
    elif r > 180 and g < 100 and b < 100:
        return "Suggested to use Cinnabar or Red Ochre, suitable for lips, ritual tools, and floral detailing."
    elif r > 180 and g > 120 and b < 100:
        return "Advised to use Yellow Ochre or Loess pigments, for skin tones, pottery, or backgrounds."
    elif r > 220 and g > 180 and b < 120:
        return "Recommended to use Realgar or Orpiment mineral pigments, suitable for edges or decorative areas."
    elif g > 150 and r < 100 and b < 120:
        return "Suggested to use Malachite Green or Mineral Green, commonly applied to plants and clothing illustration."
    elif g > 150 and b > 150 and r < 120:
        return "Recommended to use Chalcanthite or Azurite, suitable for water surfaces or cool-toned areas."
    elif b > 160 and r < 100 and g < 140:
        return "Recommended to use Ultramarine, Cobalt Blue, or Lapis Lazuli pigments for sky or fabric coloring."
    elif r > 120 and b > 120 and g < 100:
        return "Suggested to use Amethyst or Manganese Violet minerals, ideal for divine garments or ornamental regions."
    elif r > 200 and g < 140 and b > 140:
        return "Advised to use Rose Red or Tourmaline-based pigments, perfect for petals or soft detailed rendering."
    else:
        return "No clear category matched. Recommended to use Ochre-based pigments or consider color blending."