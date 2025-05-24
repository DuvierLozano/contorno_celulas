from torch.utils.data import Dataset
from PIL import Image
import os
import numpy as np
import torch as t
import segmentation_models_pytorch as smp

class load_process(Dataset):
    def __init__(self, img_path, msk_path, bb_name, idx):
        self.img_path = img_path
        self.msk_path = msk_path
        self.process = smp.encoders.get_preprocessing_fn(bb_name)
        self.lst_img = sorted(os.listdir(img_path))
        self.lst_msk = sorted(os.listdir(msk_path))
        self.idx = idx

    def __len__(self):
        return 1

    def __getitem__(self, i):
        img_full = os.path.join(self.img_path, self.lst_img[self.idx])
        msk_full = os.path.join(self.msk_path, self.lst_msk[self.idx])

        img = Image.open(img_full).convert('RGB').resize((512, 512))
        msk = Image.open(msk_full).convert('L').resize((512, 512))

        arr = np.array(img)
        m_arr = np.array(msk)

        x = t.tensor(self.process(arr), dtype=t.float32).permute(2, 0, 1)
        y = t.tensor(m_arr, dtype=t.float32).unsqueeze(0) / 255.0
        return x, y