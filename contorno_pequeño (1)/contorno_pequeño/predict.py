import os
import numpy as np
import torch as t
from PIL import Image
import segmentation_models_pytorch as smp
from utils import cerrar_bordes

def save_prediction(model, img_path, img_name, pred_path, encoder, device):
    processor = smp.encoders.get_preprocessing_fn(encoder)
    img_file = os.path.join(img_path, img_name)
    img = Image.open(img_file).convert('RGB').resize((512, 512))
    arr = np.array(img)

    inp = t.tensor(processor(arr), dtype=t.float32).permute(2, 0, 1).unsqueeze(0).to(device)
    with t.no_grad():
        pred = t.sigmoid(model(inp)).cpu().squeeze().numpy()

    pred_mask = (pred > 0.5).astype(np.uint8) * 255
    out_img = Image.fromarray(pred_mask).convert('L')
    out_img = cerrar_bordes(out_img)
    os.makedirs(pred_path, exist_ok=True)
    out_img.save(os.path.join(pred_path, f"pred_{img_name}"))
