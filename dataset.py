import os
import warnings

import pandas as pd
from PIL import Image, ImageFile
from torch.utils.data import Dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True
warnings.filterwarnings("ignore")


class FocusDataset(Dataset):
    """Image Quality Dataset."""

    def __init__(self, csv_file, root_dir, transform):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.frame = pd.read_csv(csv_file, header=None)
        self.root_dir = root_dir
        self.transform = transform

        name = self.root_dir.strip("/").split('/')[-1].lower()
        if name in ['ImageDatabasePatches2'.lower(), 'LIVE_Release2'.lower()]:
            pass
        elif name == 'FocusPath_full'.lower():
            for idx in range(len(self.frame)):
                self.frame.iloc[idx, 0] = self.frame.iloc[idx, 0][:-4] + ".png"
        elif name in ['FocusPath'.lower(), 'Corwyn_demo'.lower()]:
            for idx in range(len(self.frame)):
                self.frame.iloc[idx, 0] = self.frame.iloc[idx, 0] + ".png"
        else:
            raise Exception("Cannot not handle dataset %s" % root_dir)

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):

        img_name = os.path.join(self.root_dir, self.frame.iloc[idx, 0])

        image = Image.open(img_name)
        image = self.transform(image)

        score = abs(self.frame.iloc[idx, -1])

        sample = {'image': image, 'score': score, 'image_name': img_name, 'patch_num': image.shape[0]}

        return sample
