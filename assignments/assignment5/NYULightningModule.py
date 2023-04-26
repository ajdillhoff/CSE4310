import os

import pytorch_lightning as pl
from PIL import Image
from torch.utils.data import random_split, DataLoader
from torchvision import transforms

from NYUDataset import NYUDataset

class NYUDataModule(pl.LightningDataModule):
    """
    NYU Dataset
    Website: https://jonathantompson.github.io/NYU_Hand_Pose_Dataset.htm
    """

    def __init__(self, data_dir: str = './',
                 batch_size=32, num_workers=1,
                 img_size=224):
        """
        Args:
            data_dir (string): Path to the data.
            batch_size (int): # of samples in each batch.
            shuffle (bool): Shuffle dataset.
            num_workers (int): Number of loader threads.
            img_size (int): Size of image.
        """

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.img_size = img_size
        self.transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((img_size, img_size), transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

        super().__init__()

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        train_path = os.path.join(self.data_dir, "train/")
        test_path = os.path.join(self.data_dir, "test/")
        full = NYUDataset(train_path, self.transforms,
                          self.transforms, [], True)
        self.test = NYUDataset(test_path, self.transforms,
                               self.transforms, [], False)
        self.train, self.val = random_split(full, [69120, 3637])

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size,
                          shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers)