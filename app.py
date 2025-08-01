import streamlit as st
from PIL import Image
import json
import torch
import os
import numpy as np
import torchvision.transforms as T
import cv2
from math import sqrt
from segmentation_model import SimpleUNet
from env_suggestions import generate_environment_suggestion
from pigment_recommender import get_dominant_color, recommend_pigment

model = SimpleUNet()
model.load_state_dict(torch.load(
    "C:/Users/tv2fp3/Documents/Dunhuang_Restoration/mural_seg_model.pth",
    map_location=torch.device("cpu")
))
model.eval()

def predict_damage_mask(image, model):
    transform = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor()
    ])
    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)
        pred_mask = torch.sigmoid(output)[0][0].numpy()
        bin_mask = (pred_mask > 0.5).astype("uint8") * 255
        return Image.fromarray(bin_mask)

def simulate_restoration(image, mask, color):
    if not isinstance(image, np.ndarray):
        image = np.array(image)
    if not isinstance(mask, np.ndarray):
        mask = np.array(mask)

    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    elif image.ndim == 3 and image.shape[2] == 1:
        image = np.repeat(image, 3, axis=2)

    if mask.shape != image.shape[:2]:
        mask = np.array(Image.fromarray(mask).resize(image.shape[:2][::-1]))

    restored = image.copy()
    damage_mask = mask > 100
    damaged_indices = np.where(damage_mask)

    H, W = image.shape[:2]
    valid_indices = (damaged_indices[0] < H) & (damaged_indices[1] < W)
    y_indices = damaged_indices[0][valid_indices]
    x_indices = damaged_indices[1][valid_indices]

    for i in range(3):
        restored[y_indices, x_indices, i] = (
            0.7 * restored[y_indices, x_indices, i] +
            0.3 * color[i]
        ).astype(np.uint8)

    return Image.fromarray(restored)

def smart_inpaint_restoration(image, mask):
    image_np = np.array(image.convert("RGB"))
    mask_np = np.array(mask.convert("L"))

    if mask_np.shape != image_np.shape[:2]:
        mask_np = np.array(Image.fromarray(mask_np).resize((image_np.shape[1], image_np.shape[0])))

    damage_mask = (mask_np > 200).astype(np.uint8)
    restored_np = cv2.inpaint(image_np, damage_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    return Image.fromarray(restored_np)

def calculate_damage_ratio(mask: Image.Image) -> float:
    mask_np = np.array(mask)
    damaged_pixels = np.sum(mask_np > 100)
    total_pixels = mask_np.shape[0] * mask_np.shape[1]
    return damaged_pixels / total_pixels if total_pixels > 0 else 0.0

st.title("Mural Damage Restoration System")

uploaded_file = st.file_uploader("Upload Mural Image", type=["png", "jpg", "jpeg"])
if uploaded_file:
    input_image = Image.open(uploaded_file).convert("RGB")
    st.image(input_image, caption="Original Image", use_container_width=True)

    filename_base = os.path.splitext(os.path.basename(uploaded_file.name))[0]
    label_extensions = [".png", ".jpg", ".jpeg"]
    label_path = None

    for ext in label_extensions:
        candidate = os.path.join(
            "C:/Users/tv2fp3/Documents/Dunhuang_Restoration/data/Mural_seg/test/labels",
            filename_base + ext
        )
        if os.path.exists(candidate):
            label_path = candidate
            break

    if label_path:
        label_image = Image.open(label_path).convert("RGB")
        st.image(label_image, caption="Labeling the area of damage", use_container_width=True)
        damage_mask = label_image
        resized_image = input_image.resize((256, 256))
    else:
        resized_image = input_image.resize((256, 256))
        damage_mask = predict_damage_mask(resized_image, model)
        st.image(damage_mask, caption="Identified areas of damage", use_container_width=True)

    dominant_color = get_dominant_color(resized_image, damage_mask)
    # st.markdown(f"<b>主色 RGB：</b> {dominant_color}", unsafe_allow_html=True)

    pigment_advice = recommend_pigment(dominant_color)
    

    restored_image = smart_inpaint_restoration(resized_image, damage_mask)
    st.markdown("<b>Smart Restoration Result:</b>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.image(resized_image, caption="Original Image (256×256)", use_container_width=True)
    with col2:
        st.image(restored_image, caption="Restored Image", use_container_width=True)

    damage_ratio = calculate_damage_ratio(damage_mask)
    st.markdown(f"<b>Damaged Area Ratio:</b> {damage_ratio * 100:.2f}%", unsafe_allow_html=True)


    with open("C:/Users/tv2fp3/Documents/Dunhuang_Restoration/color_database.json", "r", encoding="utf-8") as f:
        pigment_data = json.load(f)["pigments"]

    def find_closest_pigment(rgb):
        def color_distance(c1, c2):
            return sqrt(sum((a - b) ** 2 for a, b in zip(c1, c2)))
        return min(pigment_data, key=lambda p: color_distance(rgb, p["rgb"]))

    closest_pigment = find_closest_pigment(dominant_color)

    color_hex = "#{:02x}{:02x}{:02x}".format(*dominant_color)
    st.markdown("<b>Main Color Block Display:</b>", unsafe_allow_html=True)
    st.markdown(
        f"<div style='width:100px;height:50px;background-color:{color_hex};border:1px solid #333'></div>",
        unsafe_allow_html=True
    )
    st.markdown(f"<b>Recommended Pigment:</b> {pigment_advice}", unsafe_allow_html=True)
    st.write(f"Color Name: {closest_pigment['color_name']}")
    st.write(f"Color Family: {closest_pigment['color_family']}")

    st.markdown("<b>Component Composition:</b>", unsafe_allow_html=True)
    for comp in closest_pigment["components"]:
        st.write(f"• {comp['material']} ({comp['percentage']}%)")

    st.markdown("<b>Color Mixing Suggestions:</b>", unsafe_allow_html=True)
    for mix in closest_pigment["recommended_mix"]:
        ratio_text = ', '.join([f"{k}: {v}" for k, v in mix["ratio"].items()])
        st.write(f"Target Color: {mix['target_color']}")
        st.write(f"Mixing Ratio: {ratio_text}")
        st.write(f"Application Scenario: {mix['application']}")
        st.markdown("---")