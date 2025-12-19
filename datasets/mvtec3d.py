import os
import glob
from enum import Enum

import numpy as np
import pandas as pd
import cv2
import tifffile as tif
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

from perlin import generate_perlin_mask


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
DEPTH_DIR_NAME = "depth"

to_pil = transforms.ToPILImage()
to_tensor = transforms.ToTensor()


def resize_tensor(x, target_size, convert_to_tensor=False, binary=False):
    """Resize tensor to target size using bilinear interpolation."""
    if convert_to_tensor:
        x = torch.FloatTensor(x).to('cuda')

    resized = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)

    if binary:
        resized[resized > 0] = 1

    return resized


def min_max_normalize(scores):
    """Normalize scores to [0, 1] range using min-max normalization."""
    min_val = scores.min()
    max_val = scores.max()
    normalized_scores = (scores - min_val) / (max_val - min_val)
    return normalized_scores


class DatasetSplit(Enum):
    """Dataset split types."""
    TRAIN = "train"
    TEST = "test"


def fill_missing_depth_values(depth_map, iterations=2):
    """Fill missing values (zeros) in depth map using neighbor interpolation."""
    filled_depth = depth_map.copy()

    for _ in range(iterations):
        # Create mask for zero values
        zero_mask = np.where(filled_depth == 0, 1.0, 0.0)

        depth_tensor = torch.from_numpy(filled_depth)
        height, width = depth_tensor.shape

        # Reshape for unfold operation
        depth_tensor = depth_tensor[None, None, :, :]

        # Extract 3x3 patches
        patches = torch.nn.functional.unfold(depth_tensor, 3, dilation=1, padding=1, stride=1)  # (B, C*3*3, H*W)

        # Calculate sum of non-zero values
        non_zero_mask = torch.where(patches > 0, 1.0, 0.0)
        patch_sums = torch.sum(patches, dim=1, keepdim=True)  # (B, 1, H*W)
        non_zero_counts = torch.sum(non_zero_mask, dim=1, keepdim=True)  # (B, 1, H*W)

        # Compute interpolated values
        interpolated_values = patch_sums / (non_zero_counts + 1e-12)  # (B, 1, H*W)

        # Fold back to image
        filled_patches = torch.nn.functional.fold(
            interpolated_values, (height, width), 1, dilation=1, padding=0, stride=1
        )

        # Update depth map only where zero values exist
        filled_depth = filled_patches.numpy()[0, 0] * zero_mask + (1.0 - zero_mask) * filled_depth

    return filled_depth


def refine_foreground_mask(foreground_mask):
    """Apply morphological operations to refine foreground mask."""
    kernel = np.ones((3, 3), np.uint8)
    foreground_mask[..., 0] = cv2.morphologyEx(
        foreground_mask, cv2.MORPH_CLOSE, kernel, iterations=3
    )
    return foreground_mask


def extract_foreground_mask(depth_image):
    """Extract foreground mask from depth image by detecting planar background."""
    height, width, channels = depth_image.shape
    points = np.reshape(depth_image, (height * width, channels))

    # Sample 4 corner regions to estimate background plane
    corner_size = 3
    top_left = np.sum(depth_image[:corner_size, :corner_size, :], axis=(0, 1))
    top_left_valid = np.sum(depth_image[:corner_size, :corner_size, 2] != 0)
    p1 = top_left / (top_left_valid + 1e-12)

    top_right = np.sum(depth_image[:corner_size, -corner_size:, :], axis=(0, 1))
    top_right_valid = np.sum(depth_image[:corner_size, -corner_size:, 2] != 0)
    p2 = top_right / (top_right_valid + 1e-12)

    bottom_left = np.sum(depth_image[-corner_size:, :corner_size, :], axis=(0, 1))
    bottom_left_valid = np.sum(depth_image[-corner_size:, :corner_size, 2] != 0)
    p3 = bottom_left / (bottom_left_valid + 1e-12)

    bottom_right = np.sum(depth_image[-corner_size:, -corner_size:, :], axis=(0, 1))
    bottom_right_valid = np.sum(depth_image[-corner_size:, -corner_size:, 2] != 0)
    p4 = bottom_right / (bottom_right_valid + 1e-12)

    # Estimate background plane
    plane_params = fit_plane_from_points(p1, p2, p3)
    point_distances = compute_point_plane_distances(points, np.array(plane_params))

    # Create foreground mask (points far from plane)
    threshold = 0.005
    mask = np.where(point_distances > threshold, 1.0, 0.0)
    foreground_mask = np.reshape(mask, (height, width, 1))

    return foreground_mask


def fit_plane_from_points(p1, p2, p3):
    """Fit a plane from three 3D points and return plane parameters (a, b, c, d)."""
    vector1 = p3 - p1
    vector2 = p2 - p1

    # Normal vector is cross product
    normal = np.cross(vector1, vector2)
    a, b, c = normal

    # Compute d parameter
    d = np.dot(normal, p3)

    return a, b, c, d


def compute_point_plane_distances(points, plane_params):
    """Compute distances from points to plane defined by plane_params (a, b, c, d)."""
    plane_expanded = np.expand_dims(plane_params, 0)
    distances = np.abs(
        np.sum(points * plane_expanded[:, :-1], axis=1) - plane_params[-1]
    ) / np.sum(plane_params[:-1] ** 2) ** 0.5
    return distances


class MVTec3dDataset(torch.utils.data.Dataset):
    """MVTec 3D anomaly detection dataset with RGB and depth data."""

    def __init__(
        self,
        data_root,
        anomaly_source_path="/root/dataset/dtd/images",
        dataset_name="mvtec3d",
        class_name="cable_gland",
        resize_size=288,
        crop_size=288,
        split=DatasetSplit.TRAIN,
        rotation_degree=0,
        translation_ratio=0.0,
        brightness_factor=0.0,
        contrast_factor=0.0,
        saturation_factor=0.0,
        grayscale_prob=0.0,
        hflip_prob=0.0,
        vflip_prob=0.0,
        anomaly_blend_mean=0.5,
        anomaly_blend_std=0.1,
        use_foreground_mask=False,
        use_random_augment=True,
        scale_range=0.0,
        batch_size=8,
        **kwargs
    ):
        """Initialize MVTec3D dataset.

        Args:
            data_root: Root directory of MVTec dataset.
            anomaly_source_path: Path to anomaly source images (DTD dataset).
            dataset_name: Name of the dataset.
            class_name: MVTec class name.
            resize_size: Initial resize dimension.
            crop_size: Center crop dimension.
            split: Dataset split (TRAIN or TEST).
            rotation_degree: Random rotation degree range.
            translation_ratio: Random translation ratio.
            brightness_factor: Brightness augmentation factor.
            contrast_factor: Contrast augmentation factor.
            saturation_factor: Saturation augmentation factor.
            grayscale_prob: Probability of converting to grayscale.
            hflip_prob: Probability of horizontal flip.
            vflip_prob: Probability of vertical flip.
            anomaly_blend_mean: Mean for anomaly blending factor.
            anomaly_blend_std: Standard deviation for anomaly blending factor.
            use_foreground_mask: Whether to use foreground masks.
            use_random_augment: Whether to use random augmentations.
            scale_range: Random scale range.
            batch_size: Batch size.
        """
        super().__init__()
        self.data_root = data_root
        self.split = split
        self.batch_size = batch_size
        self.anomaly_blend_mean = anomaly_blend_mean
        self.anomaly_blend_std = anomaly_blend_std
        self.use_foreground_mask = use_foreground_mask
        self.use_random_augment = use_random_augment
        self.resize_size = resize_size
        self.crop_size = crop_size
        self.class_name = class_name

        # Special handling for certain classes
        if self.class_name in ["toothbrush", "wood"]:
            self.resize_size = 329
            self.crop_size = 288

        self.image_shape = (3, self.crop_size, self.crop_size)
        self.dataset_name = dataset_name
        self.is_first_load = True
        self.class_foreground_ratio = use_foreground_mask

        # Load image and depth paths
        self.image_paths_per_class, self.image_metadata, self.depth_paths_per_class, self.depth_paths = self._load_image_paths()

        # Load anomaly source images
        self.anomaly_source_images = sorted(glob.glob(anomaly_source_path + "/*/*.jpg"))

        # Image transformations
        self.image_transform = transforms.Compose([
            transforms.Resize(self.resize_size),
            transforms.ColorJitter(brightness_factor, contrast_factor, saturation_factor),
            transforms.RandomHorizontalFlip(hflip_prob),
            transforms.RandomVerticalFlip(vflip_prob),
            transforms.RandomGrayscale(grayscale_prob),
            transforms.RandomAffine(
                rotation_degree,
                translate=(translation_ratio, translation_ratio),
                scale=(1.0 - scale_range, 1.0 + scale_range),
                interpolation=transforms.InterpolationMode.BILINEAR
            ),
            transforms.CenterCrop(self.crop_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

        # Mask transformations
        self.mask_transform = transforms.Compose([
            transforms.Resize(self.resize_size),
            transforms.CenterCrop(self.crop_size),
            transforms.ToTensor(),
        ])

    def rand_augmenter(self):
        """Create randomized augmentation pipeline."""
        augmentations = [
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

        selected_indices = np.random.choice(len(augmentations), 3, replace=False)

        augment_pipeline = [
            transforms.Resize(self.resize_size),
            augmentations[selected_indices[0]],
            augmentations[selected_indices[1]],
            augmentations[selected_indices[2]],
            transforms.CenterCrop(self.crop_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]

        return transforms.Compose(augment_pipeline)

    def load_depth_data(self, index):
        """Load and preprocess depth data for given index."""
        raw_depth = tif.imread(self.depth_paths[index]).astype(np.float32)
        depth_3channel = np.array(raw_depth).reshape((raw_depth.shape[0], raw_depth.shape[1], 3)).astype(np.float32)

        # Extract depth from last channel
        depth_map = depth_3channel[:, :, 2]
        missing_data_mask = np.where(depth_map == 0, 1.0, 0.0)

        # Get foreground mask by plane detection
        foreground_mask = extract_foreground_mask(depth_3channel)
        foreground_mask[..., 0] = foreground_mask[..., 0] * (1.0 - missing_data_mask)
        foreground_mask = refine_foreground_mask(foreground_mask)

        # Apply morphological operations
        kernel = np.ones((5, 5), np.uint8)
        foreground_mask = cv2.dilate(foreground_mask, kernel, iterations=1)
        foreground_mask = cv2.erode(foreground_mask, kernel, iterations=1)

        # Apply mask and normalize depth
        depth_map = depth_map * foreground_mask
        missing_data_mask = np.where(depth_map == 0, 1.0, 0.0)

        max_depth = np.max(depth_map)
        min_depth = np.min(depth_map * (1.0 - missing_data_mask) + 1000 * missing_data_mask)
        depth_map = (depth_map - min_depth) / (max_depth - min_depth)

        # Fill missing depth values
        depth_map = fill_missing_depth_values(depth_map)

        # Create binary foreground mask
        binary_foreground = 1.0 - missing_data_mask

        return depth_map, binary_foreground

    def normalize_depth_tensor(self, depth_data, target_size, binary=False):
        """Normalize depth tensor to target size."""
        depth_data = depth_data.copy()
        depth_tensor = torch.FloatTensor(depth_data)

        if len(depth_tensor.shape) == 2:
            depth_tensor = depth_tensor[None, None]
            channels = 1
        elif len(depth_tensor.shape) == 3:
            depth_tensor = depth_tensor.permute(2, 0, 1)[None]
            channels = depth_tensor.shape[1]
        else:
            raise ValueError(f"Invalid depth tensor shape: {depth_tensor.shape}")

        depth_tensor = resize_tensor(depth_tensor, (target_size, target_size), binary=binary)
        depth_tensor = depth_tensor.reshape(channels, target_size, target_size)
        return depth_tensor

    def __getitem__(self, idx):
        """Get dataset item at index."""
        # Load and normalize depth data
        depth_map, foreground_mask = self.load_depth_data(idx)
        depth_tensor = self.normalize_depth_tensor(depth_map, self.crop_size, binary=False)
        foreground_tensor = self.normalize_depth_tensor(foreground_mask, self.crop_size, binary=True).squeeze(0)

        # Load and preprocess RGB image
        class_name, anomaly_type, image_path, ground_truth_mask_path = self.image_metadata[idx]
        rgb_image = Image.open(image_path).convert("RGB")
        rgb_image = self.image_transform(rgb_image)

        # Repeat depth to 3 channels and apply image transforms
        depth_tensor = depth_tensor.repeat(3, 1, 1)
        depth_tensor = self.image_transform(to_pil(depth_tensor))

        # Initialize default tensors
        augmented_rgb = augmented_depth = mask_small = mask_small_0 = mask_small_1 = depth_tensor

        if self.split == DatasetSplit.TRAIN:
            # Load anomaly source images
            anomaly_source = Image.open(np.random.choice(self.anomaly_source_images)).convert("RGB")
            anomaly_source_gray = anomaly_source.convert("L").convert("RGB")

            # Apply random augmentation
            if self.use_random_augment:
                augment_transform = self.rand_augmenter()
                anomaly_source = augment_transform(anomaly_source)
                anomaly_source_gray = augment_transform(anomaly_source_gray)
            else:
                anomaly_source = self.image_transform(anomaly_source)
                anomaly_source_gray = self.image_transform(anomaly_source_gray)

            # Load or compute foreground mask
            final_foreground_mask = foreground_tensor
            if self.use_foreground_mask == 2:
                xlsx_path = f"./datasets/excel/{self.dataset_name}_distribution.xlsx"
                try:
                    if self.is_first_load:
                        df = pd.read_excel(xlsx_path)
                        class_key = f"{self.dataset_name}_{class_name}"
                        self.class_foreground_ratio = df.loc[df['Class'] == class_key, 'Foreground'].values[0]
                        self.is_first_load = False
                except:
                    self.class_foreground_ratio = 1.0
            elif self.use_foreground_mask == 1:
                self.class_foreground_ratio = 1.0
            else:
                self.class_foreground_ratio = 0.0

            if self.class_foreground_ratio > 0:
                foreground_mask_path = image_path.replace(class_name, "fg_mask/" + class_name)
                loaded_mask = Image.open(foreground_mask_path)
                loaded_mask_binary = torch.ceil(self.mask_transform(loaded_mask)[0])
                final_foreground_mask = torch.maximum(foreground_tensor, loaded_mask_binary)

            # Generate Perlin noise masks
            mask_all = generate_perlin_mask(
                rgb_image.shape, self.crop_size // 8, 0, 6, final_foreground_mask, 1
            )
            mask_small, mask_large = torch.from_numpy(mask_all[0]), torch.from_numpy(mask_all[1])

            mask_all_0 = generate_perlin_mask(
                rgb_image.shape, self.crop_size // 8, 0, 6, final_foreground_mask, 1
            )
            mask_small_0, mask_large_0 = torch.from_numpy(mask_all_0[0]), torch.from_numpy(mask_all_0[1])

            mask_all_1 = generate_perlin_mask(
                rgb_image.shape, self.crop_size // 8, 0, 6, final_foreground_mask, 1
            )
            mask_small_1, mask_large_1 = torch.from_numpy(mask_all_1[0]), torch.from_numpy(mask_all_1[1])

            # Apply anomaly augmentation
            beta = np.random.normal(self.anomaly_blend_mean, self.anomaly_blend_std)
            beta = np.clip(beta, 0.2, 0.8)

            augmentation_type = np.random.rand()
            if augmentation_type > 0.66:
                # Augment both RGB and depth
                augmented_rgb = rgb_image * (1 - mask_large) + (1 - beta) * anomaly_source * mask_large + beta * rgb_image * mask_large
                augmented_depth = depth_tensor * (1 - mask_large) + (1 - beta) * anomaly_source_gray * mask_large + beta * depth_tensor * mask_large
            elif augmentation_type > 0.33:
                # Augment only RGB
                augmented_rgb = rgb_image * (1 - mask_large) + (1 - beta) * anomaly_source * mask_large + beta * rgb_image * mask_large
                augmented_depth = depth_tensor
            else:
                # Augment only depth
                augmented_rgb = rgb_image
                augmented_depth = depth_tensor * (1 - mask_large) + (1 - beta) * anomaly_source_gray * mask_large + beta * depth_tensor * mask_large

        # Load ground truth mask for test images
        if self.split == DatasetSplit.TEST and ground_truth_mask_path is not None:
            ground_truth_mask = Image.open(ground_truth_mask_path).convert('L')
            ground_truth_mask = self.mask_transform(ground_truth_mask)
        else:
            ground_truth_mask = torch.zeros([1, *rgb_image.size()[1:]])

        return {
            "image": rgb_image,
            "aug_image": augmented_rgb,
            "depth": depth_tensor,
            "aug_depth": augmented_depth,
            "mask_s": mask_small,
            "mask_s_0": mask_small_0,
            "mask_s_1": mask_small_1,
            "mask_gt": ground_truth_mask,
            "is_anomaly": int(anomaly_type != "good"),
            "image_path": image_path,
        }

    def __len__(self):
        """Return dataset length."""
        return len(self.image_metadata)

    def _load_image_paths(self):
        """Load all image, depth, and mask paths from dataset directory."""
        image_paths = {}
        depth_paths_dict = {}
        mask_paths = {}

        image_dir = os.path.join(self.data_root, self.class_name, self.split.value)
        depth_dir = os.path.join(self.data_root, DEPTH_DIR_NAME, self.class_name, self.split.value)
        mask_dir = os.path.join(self.data_root, self.class_name, "ground_truth")

        anomaly_types = os.listdir(image_dir)

        image_paths[self.class_name] = {}
        depth_paths_dict[self.class_name] = {}
        mask_paths[self.class_name] = {}

        for anomaly_type in anomaly_types:
            image_anomaly_dir = os.path.join(image_dir, anomaly_type)
            image_files = sorted(os.listdir(image_anomaly_dir))
            image_paths[self.class_name][anomaly_type] = [os.path.join(image_anomaly_dir, f) for f in image_files]

            depth_anomaly_dir = os.path.join(depth_dir, anomaly_type)
            depth_files = sorted(os.listdir(depth_anomaly_dir))
            depth_paths_dict[self.class_name][anomaly_type] = [os.path.join(depth_anomaly_dir, f) for f in depth_files]

            if self.split == DatasetSplit.TEST and anomaly_type != "good":
                mask_anomaly_dir = os.path.join(mask_dir, anomaly_type)
                mask_files = sorted(os.listdir(mask_anomaly_dir))
                mask_paths[self.class_name][anomaly_type] = [os.path.join(mask_anomaly_dir, f) for f in mask_files]
            else:
                mask_paths[self.class_name]["good"] = None

        # Create metadata list for iteration
        image_metadata = []
        for class_name in sorted(image_paths.keys()):
            for anomaly_type in sorted(image_paths[class_name].keys()):
                for i, image_path in enumerate(image_paths[class_name][anomaly_type]):
                    metadata = [class_name, anomaly_type, image_path]
                    if self.split == DatasetSplit.TEST and anomaly_type != "good":
                        metadata.append(mask_paths[class_name][anomaly_type][i])
                    else:
                        metadata.append(None)
                    image_metadata.append(metadata)

        # Collect all depth paths
        all_depth_paths = []
        for class_name in sorted(depth_paths_dict.keys()):
            for anomaly_type in sorted(depth_paths_dict[class_name].keys()):
                all_depth_paths.extend(depth_paths_dict[class_name][anomaly_type])

        return image_paths, image_metadata, depth_paths_dict, all_depth_paths
