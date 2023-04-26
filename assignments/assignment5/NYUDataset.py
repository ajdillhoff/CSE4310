import os

import torch
import scipy.io as sio
import numpy as np
from PIL import Image


def normalize(points, center, scale_factor):
    """
    Normalize a set of points centered on `center` and scaled by
    `scale_factor`.

    Args:
        center (array): Center of object
        scale_factor (float): Scale factor.
    Returns:
        Normalized points.
    """
    norm_points = points.clone()
    norm_points -= center
    norm_points *= scale_factor

    return norm_points.to(torch.float32)


def denormalize(points, scale_factor, center=None):
    points /= scale_factor
    if center is not None:
        points += center

    return points


class NYUDataset(torch.utils.data.Dataset):
    """NYU Dataset."""

    def __init__(self, root_dir, sample_transform,
                 target_transform, idxs=[], train=False):
        """
        Args:
            root_dir (string): Path to the data.
            sample_transform (callable, optional): Optional transform to be
                applied to the sample.
            target_transform (callable, optional): Optional transform to be
                applied to the target.
            num_points (int, optional): Number of points to sample in the
                point cloud.
        """
        self.root_dir = root_dir
        self.sample_transform = sample_transform
        self.target_transform = target_transform
        self.num_kp = 22
        self.train = train
        self.img_pad = 10
        if len(idxs) == 0:
            if not train:
                idxs = list([i for i in range(8252)])
            else:
                idxs = list([i for i in range(72757)])
        self.idxs = idxs

        # Load annotation file
        anno_mat = sio.loadmat(os.path.join(self.root_dir, "joint_data.mat"))
        self.annotations2d = anno_mat['joint_uvd'][0]
        self.annotations3d = anno_mat['joint_xyz'][0]
        self.eval_joints = [0, 3, 6, 9, 12, 15, 18, 21, 24, 25, 27, 30, 31, 32]
        self.joint_idxs = [0, 1, 3, 5, 6, 7, 9, 11, 12, 13, 15, 17, 18, 19, 21, 23, 24, 25, 26, 28, 35, 32]
        self.nyu_to_model_idxs = [20, 19, 18, 17, 16, 3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12, 21]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, idx):
        idx = self.idxs[idx] + 1
        file_name = os.path.join(self.root_dir,
                                 'rgb_1_{0:07d}.png'.format(idx))

        image = np.asarray(Image.open(file_name))

        kps2d = self.annotations2d[idx-1, self.joint_idxs, :2]

        # Crop the hand from the image using kps2d
        x_min = kps2d[:, 0].min() - self.img_pad
        x_max = kps2d[:, 0].max() + self.img_pad
        y_min = kps2d[:, 1].min() - self.img_pad
        y_max = kps2d[:, 1].max() + self.img_pad
        x_diff = x_max - x_min
        y_diff = y_max - y_min
        if x_diff > y_diff:
            y_min -= (x_diff - y_diff) / 2
            y_max += (x_diff - y_diff) / 2
        else:
            x_min -= (y_diff - x_diff) / 2
            x_max += (y_diff - x_diff) / 2
        center = torch.tensor([x_min, y_min])
        x_min = int(max(0, x_min))
        x_max = int(min(image.shape[1], x_max))
        y_min = int(max(0, y_min))
        y_max = int(min(image.shape[0], y_max))
        image = image[y_min:y_max, x_min:x_max]

        # Normalize the keypoints to match the cropped image
        scale_factor = 224 / image.shape[0]
        target_pix = normalize(torch.tensor(kps2d), center, scale_factor)

        # Get center of mass of the hand
        com = kps2d[21]

        # Get the scale factor
        sf = 1.0 / (kps2d.max(0)[0] - kps2d.min(0)[0])

        # Normalize the keypoints to be in the range [-1, 1]
        target = normalize(torch.tensor(kps2d), com, sf)

        if self.sample_transform:
            image = self.sample_transform(image)

        return image, target, target_pix, center, scale_factor