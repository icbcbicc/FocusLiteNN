import os
import warnings

import pandas as pd
from PIL import Image, ImageFile
from torch.utils.data import Dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True
warnings.filterwarnings("ignore")


class FocusDataset(Dataset):

    def __init__(self, csv_file, root_dir, transform):
        self.frame = pd.read_csv(csv_file, header=None)
        self.root_dir = root_dir
        self.transform = transform

        if csv_file.find("TCGA@Focus") != -1:
            pass
        elif csv_file.find("FocusPath_full") != -1:
            for idx in range(len(self.frame)):
                self.frame.iloc[idx, 0] = self.frame.iloc[idx, 0][:-4] + ".png"
        elif csv_file.find("FocusPath") != -1:
            for idx in range(len(self.frame)):
                self.frame.iloc[idx, 0] = self.frame.iloc[idx, 0] + ".png"
        else:
            raise Exception(f"Cannot not handle dataset {root_dir}")

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):

        img_name = os.path.join(self.root_dir, self.frame.iloc[idx, 0])

        image = Image.open(img_name)
        image = self.transform(image)

        score = abs(self.frame.iloc[idx, -1])

        sample = {'image': image, 'score': score, 'image_name': img_name, 'patch_num': image.shape[0]}

        return sample
