from torchvision import transforms
from perlin import perlin_mask
from enum import Enum
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import PIL
import torch
import os
import glob
import tifffile as tif
import cv2
to_pil = transforms.ToPILImage()
to_tensor = transforms.ToTensor()

_CLASSNAMES = [
    "bagel",
    "dowel",
    "potato"
]
from PIL import Image

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
DEEP_SOURCE_VALUE = "depth"


def downsampling(x, size, to_tensor=False, bin=False):
    if to_tensor:
        x = torch.FloatTensor(x).to('cuda')
    down = F.interpolate(x, size=size, mode='bilinear', align_corners=False)
    if bin:
        down[down > 0] = 1
    return down


def min_max_normalize(scores):
    min_val = scores.min()
    max_val = scores.max()
    normalized_tensor = (scores - min_val) / (max_val - min_val)
    return normalized_tensor


class DatasetSplit(Enum):
    TRAIN = "train"
    TEST = "test"


def fill_depth_map(depth_image, iterations=2):
    depth_image = depth_image
    for i in range(iterations):
        zero_mask = np.where(depth_image == 0, np.ones_like(depth_image), np.zeros_like(depth_image))
        depth_image_tensor = torch.from_numpy(depth_image)
        h, w = depth_image_tensor.shape
        depth_image_tensor = depth_image_tensor.reshape((1, 1, h, w))  # use only depth
        depth_image_t = torch.nn.functional.unfold(depth_image_tensor, 3, dilation=1, padding=1, stride=1)  # B, 1x3x3, L -> L=HW
        depth_image_t_nonzero_sum = torch.sum(torch.where(depth_image_t > 0, torch.ones_like(depth_image_t), torch.zeros_like(depth_image_t)),
                                       dim=1,
                                       keepdim=True)
        depth_image_t_sum = torch.sum(depth_image_t, dim=1, keepdim=True)  # B, 1, L
        depth_image_t_filtered = depth_image_t_sum / (depth_image_t_nonzero_sum + 1e-12)
        depth_image_out = torch.nn.functional.fold(depth_image_t_filtered, depth_image.shape[:2], 1, dilation=1, padding=0,
                                            stride=1)  # B, 1, H, W
        depth_image = depth_image_out.numpy()[0, 0, :, :] * zero_mask + (1.0 - zero_mask) * depth_image
    return depth_image


def fill_plane_mask(plane_mask):
    kernel = np.ones((3, 3), np.uint8)
    plane_mask[:, :, 0] = cv2.morphologyEx(plane_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    return plane_mask


def get_plane_mask(depth_image):
    h, w, c = depth_image.shape  # 800，800，3
    points = np.reshape(depth_image, (h * w, c))  # （640000,3）
    p1 = np.sum(depth_image[:3, :3, :], axis=(0, 1)) / (np.sum(depth_image[:3, :3, 2] != 0) + 1e-12)
    p2 = np.sum(depth_image[:3, -3:, :], axis=(0, 1)) / (np.sum(depth_image[:3, -3:, 2] != 0) + 1e-12)
    p3 = np.sum(depth_image[-3:, :3, :], axis=(0, 1)) / (np.sum(depth_image[-3:, :3, 2] != 0) + 1e-12)
    p4 = np.sum(depth_image[-3:, -3:, :], axis=(0, 1)) / (np.sum(depth_image[-3:, -3:, 2] != 0) + 1e-12)
    plane = get_plane_from_points(p1, p2, p3)
    point_distance = get_distance_to_plane(points, np.array(plane))
    points_mask = np.where(point_distance > 0.005, np.ones_like(point_distance), np.zeros_like(point_distance))
    plane_mask = np.reshape(points_mask, (h, w, 1))
    return plane_mask


def get_plane_from_points(p1, p2, p3):
    v1 = p3 - p1
    v2 = p2 - p1

    cp = np.cross(v1, v2)
    a, b, c = cp

    d = np.dot(cp, p3)

    return a, b, c, d


def get_distance_to_plane(points, plane):
    plane_rs = np.expand_dims(plane, 0)
    dist = np.abs(np.sum(points * plane_rs[:, :-1], axis=1) - plane[-1]) / np.sum(plane[:-1] ** 2) ** 0.5
    return dist


class MVTec3dDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for MVTec.
    """

    def __init__(
            self,
            source,
            anomaly_source_path='/root/dataset/dtd/images',
            dataset_name='mvtec3d',
            classname='leather',
            resize=288,
            imagesize=288,
            split=DatasetSplit.TRAIN,
            rotate_degrees=0,
            translate=0,
            brightness_factor=0,
            contrast_factor=0,
            saturation_factor=0,
            gray_p=0,
            h_flip_p=0,
            v_flip_p=0,
            distribution=0,
            mean=0.5,
            std=0.1,
            fg=0,
            random_augmentation=1,
            scale=0,
            batch_size=8,
            **kwargs,
    ):
        """
        Args:
            source: [str]. Path to the MVTec data folder.
            classname: [str or None]. Name of MVTec class that should be
                       provided in this dataset. If None, the datasets
                       iterates over all available images.
            resize: [int]. (Square) Size the loaded image initially gets
                    resized to.
            imagesize: [int]. (Square) Size the resized loaded image gets
                       (center-)cropped to.
            split: [enum-option]. Indicates if training or test split of the
                   data should be used. Has to be an option taken from
                   DatasetSplit, e.g. mvtec.DatasetSplit.TRAIN. Note that
                   mvtec.DatasetSplit.TEST will also load mask data.
        """
        super().__init__()
        self.source = source
        self.split = split
        self.batch_size = batch_size
        self.distribution = distribution
        self.mean = mean
        self.std = std
        self.resize = resize if self.distribution != 1 else [resize, resize]
        self.image_size = imagesize
        self.imagesize = (3, self.image_size, self.image_size)
        self.classname = classname
        self.random_augmentation = random_augmentation

        if self.distribution != 1 and (self.classname == 'toothbrush' or self.classname == 'wood'):
            self.resize = 329
            self.image_size = 288
            self.imagesize = (3, self.image_size, self.image_size)

        self.dataset_name = dataset_name
        self.first_read = 1

        self.image_paths_per_class, self.images_to_iterate, self.depth_paths_per_class, self.depths_to_iterate = self.get_image_data()
        self.anomaly_source_paths = sorted(1 * glob.glob(anomaly_source_path + "/*/*.jpg") +
                                           0 * list(next(iter(self.image_paths_per_class.values())).values())[0])

        self.transform_image = [
            transforms.Resize(self.resize),
            transforms.ColorJitter(brightness_factor, contrast_factor, saturation_factor),
            transforms.RandomHorizontalFlip(h_flip_p),
            transforms.RandomVerticalFlip(v_flip_p),
            transforms.RandomGrayscale(gray_p),
            transforms.RandomAffine(rotate_degrees,
                                    translate=(translate, translate),
                                    scale=(1.0 - scale, 1.0 + scale),
                                    interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
        self.transform_image = transforms.Compose(self.transform_image)

        self.transform_mask = [
            transforms.Resize(self.resize),
            transforms.CenterCrop(self.image_size),
            transforms.ToTensor(),
        ]
        self.transform_mask = transforms.Compose(self.transform_mask)

    def rand_augmenter(self):
        list_aug = [
            transforms.ColorJitter(contrast=(0.8, 1.2)),
            transforms.ColorJitter(brightness=(0.8, 1.2)),
            transforms.ColorJitter(saturation=(0.8, 1.2), hue=(-0.2, 0.2)),
            transforms.RandomHorizontalFlip(p=1),
            transforms.RandomVerticalFlip(p=1),
            transforms.RandomGrayscale(p=1),
            transforms.RandomAutocontrast(p=1),
            transforms.RandomEqualize(p=1),
            transforms.RandomAffine(degrees=(-45, 45)),
        ]
        aug_idx = np.random.choice(np.arange(len(list_aug)), 3, replace=False)

        transform_augmentation = [
            transforms.Resize(self.resize),
            list_aug[aug_idx[0]],
            list_aug[aug_idx[1]],
            list_aug[aug_idx[2]],
            transforms.CenterCrop(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]

        transform_augmentation = transforms.Compose(transform_augmentation)
        return transform_augmentation

    def get_3D(self, index):
        """Extract and process 3D depth data from TIFF file.

        Args:
            index: Index of the depth file to process

        Returns:
            Tuple of (processed_depth, foreground_mask)
        """
        # Read depth data from TIFF file (shape: 800x800x3)
        depth = tif.imread(self.depths_to_iterate[index]).astype(np.float32)
        image_t = np.array(depth).reshape((depth.shape[0], depth.shape[1], 3)).astype(np.float32)

        # Extract only the depth channel (third channel)
        depth = image_t[:, :, 2]

        # Create mask for zero values (unmeasured regions at object edges)
        zero_mask = np.where(depth == 0, np.ones_like(depth), np.zeros_like(depth))

        # Get plane mask to identify foreground objects
        plane_mask = get_plane_mask(image_t)

        # Remove unmeasured data from plane mask
        plane_mask[:, :, 0] = plane_mask[:, :, 0] * (1.0 - zero_mask)

        # Apply morphological operations to smooth the mask
        plane_mask = fill_plane_mask(plane_mask)
        kernel = np.ones((5, 5), np.uint8)
        plane_mask = cv2.dilate(plane_mask, kernel, iterations=1)
        plane_mask = cv2.erode(plane_mask, kernel, iterations=1)
        plane_mask = cv2.dilate(plane_mask, kernel, iterations=1)
        plane_mask = cv2.erode(plane_mask, kernel, iterations=1)

        # Apply plane mask to keep only foreground depth data
        depth = depth * plane_mask

        # Update zero mask after plane masking (foreground=0, background=1)
        zero_mask = np.where(depth == 0, np.ones_like(depth), np.zeros_like(depth))

        # Normalize depth values to [0, 1] range
        im_max = np.max(depth)
        im_min = np.min(depth * (1.0 - zero_mask) + 1000 * zero_mask)
        depth = (depth - im_min) / (im_max - im_min)
        depth = depth * (1.0 - zero_mask)

        # Fill holes in depth map
        depth = fill_depth_map(depth)

        # Create foreground mask (inverse of zero mask)
        fg = 1 - zero_mask

        return depth, fg

    def transform_3D(self, x, img_len, binary=False):
        """Transform 3D data to specified size.

        Args:
            x: Input data (2D or 3D array)
            img_len: Target image length
            binary: Whether to apply binary thresholding

        Returns:
            Transformed tensor of shape (channels, img_len, img_len)
        """
        x = x.copy()
        x = torch.FloatTensor(x)

        # Handle different input dimensions
        if len(x.shape) == 2:
            x = x[None, None]
            channels = 1
        elif len(x.shape) == 3:
            x = x.permute(2, 0, 1)[None]
            channels = x.shape[1]
        else:
            raise Exception(f'invalid dimensions of x:{x.shape}')

        # Downsample to target size
        x = downsampling(x, (img_len, img_len), bin=binary)
        x = x.reshape(channels, img_len, img_len)
        return x

    def __getitem__(self, idx):
        depth = torch.tensor([1])
        fg = torch.tensor([1])
        depth, depth_foreground = self.get_3D(idx)
        depth = self.transform_3D(depth, self.image_size, binary=False)
        mask = depth.squeeze()
        mask = np.clip((mask * 500), 0, 255)
        image = Image.fromarray(mask.numpy().astype(np.uint8))
        depth_foreground = self.transform_3D(depth_foreground, self.image_size, binary=True)
        depth_foreground = depth_foreground.squeeze(0)

        classname, anomaly, image_path, mask_path = self.images_to_iterate[idx]
        image = PIL.Image.open(image_path).convert("RGB")
        image = self.transform_image(image)

        depth = depth.repeat(3, 1, 1)
        depth = to_pil(depth)
        depth = self.transform_image(depth)


        foreground_mask = small_mask = augmented_image = augmented_depth = torch.tensor([1])


        if self.split == DatasetSplit.TRAIN:
            # Load random augmentation image
            augmentation = PIL.Image.open(np.random.choice(self.anomaly_source_paths)).convert("RGB")

            # Convert to grayscale ('L' mode) then back to RGB
            gray_augmentation = augmentation.convert('L')
            gray_augmentation = gray_augmentation.convert('RGB')

            # Apply random augmentation transforms
            if self.random_augmentation:
                transform_augmentation = self.rand_augmenter()
                augmentation = transform_augmentation(augmentation)
                gray_augmentation = transform_augmentation(gray_augmentation)
            else:
                augmentation = self.transform_image(augmentation)
                gray_augmentation = self.transform_image(gray_augmentation)

            foreground_mask = depth_foreground

            # Generate Perlin noise masks for augmentation
            all_masks = perlin_mask(image.shape, self.image_size // 8, 0, 6, foreground_mask, 1)
            small_mask = torch.from_numpy(all_masks[0])
            large_mask = torch.from_numpy(all_masks[1])

            # Generate random blending coefficient
            beta = np.random.normal(loc=self.mean, scale=self.std)
            beta = np.clip(beta, 0.2, 0.8)

            # Randomly select augmentation strategy
            random_augmentation = np.random.rand()
            if random_augmentation > 0.66:
                # Augment both RGB and depth
                augmented_image = image * (1 - large_mask) + (1 - beta) * augmentation * large_mask + beta * image * large_mask
                augmented_depth = depth * (1 - large_mask) + (1 - beta) * gray_augmentation * large_mask + beta * depth * large_mask
            elif random_augmentation > 0.33:
                # Augment only RGB
                augmented_image = image * (1 - large_mask) + (1 - beta) * augmentation * large_mask + beta * image * large_mask
                augmented_depth = depth
            else:
                # Augment only depth
                augmented_image = image
                augmented_depth = depth * (1 - large_mask) + (1 - beta) * gray_augmentation * large_mask + beta * depth * large_mask

            

        if self.split == DatasetSplit.TEST and mask_path is not None:
            ground_truth_mask = PIL.Image.open(mask_path).convert('L')
            ground_truth_mask = self.transform_mask(ground_truth_mask)
        else:
            ground_truth_mask = torch.zeros([1, *image.size()[1:]])


        return {
            "image": image,
            "augmented_image": augmented_image,
            "depth": depth,
            "augmented_depth": augmented_depth,
            "small_mask": small_mask,
            "ground_truth_mask": ground_truth_mask,
            "is_anomaly": int(anomaly != "good"),
            "image_path": image_path,
        }

    def __len__(self):
        return len(self.images_to_iterate)

    def get_image_data(self):
        image_paths_per_class = {}
        depth_paths_per_class = {}
        mask_paths_per_class = {}

        image_path = os.path.join(self.source, self.classname, self.split.value)  # split.value 是
        depth_path = os.path.join(self.source, DEEP_SOURCE_VALUE, self.classname, self.split.value)
        mask_path = os.path.join(self.source, self.classname, "ground_truth")
        anomaly_types = os.listdir(image_path)

        image_paths_per_class[self.classname] = {}
        depth_paths_per_class[self.classname] = {}
        mask_paths_per_class[self.classname] = {}

        for anomaly in anomaly_types:
            anomaly_path = os.path.join(image_path, anomaly)
            anomaly_files = sorted(os.listdir(anomaly_path))
            image_paths_per_class[self.classname][anomaly] = [os.path.join(anomaly_path, x) for x in anomaly_files]
            anomaly_deep_path = os.path.join(depth_path, anomaly)
            anomaly_deep_files = sorted(os.listdir(anomaly_deep_path))
            depth_paths_per_class[self.classname][anomaly] = [os.path.join(anomaly_deep_path, x) for x in
                                                         anomaly_deep_files]

            if self.split == DatasetSplit.TEST and anomaly != "good":
                anomaly_mask_path = os.path.join(mask_path, anomaly)
                anomaly_mask_files = sorted(os.listdir(anomaly_mask_path))
                mask_paths_per_class[self.classname][anomaly] = [os.path.join(anomaly_mask_path, x) for x in
                                                                anomaly_mask_files]
            else:
                mask_paths_per_class[self.classname]["good"] = None

        images_to_iterate = []
        for classname in sorted(image_paths_per_class.keys()):
            for anomaly in sorted(image_paths_per_class[classname].keys()):
                for i, image_path in enumerate(image_paths_per_class[classname][anomaly]):
                    data_tuple = [classname, anomaly, image_path]
                    if self.split == DatasetSplit.TEST and anomaly != "good":
                        data_tuple.append(mask_paths_per_class[classname][anomaly][i])
                    else:
                        data_tuple.append(None)
                    images_to_iterate.append(data_tuple)

        depths_to_iterate = []
        for classname in sorted(depth_paths_per_class.keys()):
            for anomaly in sorted(depth_paths_per_class[classname].keys()):
                for i, deep_path in enumerate(depth_paths_per_class[classname][anomaly]):
                    depths_to_iterate.append(deep_path)

        return image_paths_per_class, images_to_iterate, depth_paths_per_class, depths_to_iterate
