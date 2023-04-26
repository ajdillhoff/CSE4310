# Assignment 5 (BONUS)

Finetune a pretrained a [ViT](https://arxiv.org/abs/2010.11929) model on the [NYU Hand Pose Dataset](https://jonathantompson.github.io/NYU_Hand_Pose_Dataset.htm).

## Setup

Most of the code is already provided in the `deep_learning/vit_hand_pose` located in this repository. You will just need to download the dataset. A smaller version of the dataset is linked on Canvas.

### `NYUDataModule`

The `NYUDataModule` class is already configured to load the dataset once the data has been downloaded and unzipped. Pass the directory containing the dataset to the `NYUDataModule` constructor. The dataset itself returns 5 items:
- `image`: The RGB image
- `target`: The normalized 2D joint locations (22 keypoints)
- `kps2d_pixel`: The 2D joint locations in pixel coordinates relative to the cropped image.
- `center`: The center of the hand in pixel coordinates relative to the original image.
- `scale`: The scale factor used to crop the image.

### `NYULightningModule`

The `NYULightningModule` class is already configured to use the `ViT_b_16` model from `torchvision`. The original output of this model is a `torch.Tensor` of shape `(batch_size, 1000)`. You will need to modify the `forward` method to return a tensor of shape `(batch_size, 22 * 2)` since we are predicting 2D joint locations. You will also need to modify the `training_step` method to calculate the loss between the predicted 2D joint locations and the ground truth 2D joint locations.

## Measuring Performance

The `NYULightningModule` class is already configured to calculate the mean squared error between the predicted 2D joint locations and the ground truth 2D joint locations assuming the output is flattened to a tensor of shape `(batch_size, 22 * 2)`. You will need to modify the `validation_step` method to calculate the mean squared error between the predicted 2D joint locations and the ground truth 2D joint locations in pixel space.

## Submission

When you are finished, submit your completed code along with the following:
- A pixel-wise accuracy measurement of the model on the test set BEFORE finetuning.
- A pixel-wise accuracy measurement of the model on the test set AFTER finetuning.