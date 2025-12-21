"""Perlin noise generation for data augmentation.

This module provides functions to generate Perlin noise masks for image augmentation,
commonly used in anomaly detection tasks.
"""

import imgaug.augmenters as iaa
import numpy as np
import torch
import math


def generate_threshold_mask(image_shape, min_scale=0, max_scale=4):
    """Generate a binary threshold mask using Perlin noise.

    Args:
        image_shape: Shape of the image (C, H, W)
        min_scale: Minimum scale for Perlin noise (default: 0)
        max_scale: Maximum scale for Perlin noise (default: 4)

    Returns:
        Binary threshold mask as numpy array
    """
    min_perlin_scale = min_scale
    max_perlin_scale = max_scale
    perlin_scale_x = 2 ** np.random.randint(min_perlin_scale, max_perlin_scale)
    perlin_scale_y = 2 ** np.random.randint(min_perlin_scale, max_perlin_scale)

    # Generate Perlin noise
    perlin_noise = generate_perlin_noise_2d(
        (image_shape[1], image_shape[2]),
        (perlin_scale_x, perlin_scale_y)
    )

    # Apply random rotation
    threshold = 0.5
    perlin_noise = iaa.Sequential([iaa.Affine(rotate=(-90, 90))])(image=perlin_noise)

    # Create binary threshold mask
    threshold_mask = np.where(
        perlin_noise > threshold,
        np.ones_like(perlin_noise),
        np.zeros_like(perlin_noise)
    )

    return threshold_mask


def perlin_mask(image_shape, feature_size, min_scale, max_scale, foreground_mask, return_large_mask=0):
    """Generate Perlin noise mask for data augmentation.

    Args:
        image_shape: Shape of the image (C, H, W)
        feature_size: Size of the downsampled feature map
        min_scale: Minimum scale for Perlin noise
        max_scale: Maximum scale for Perlin noise
        foreground_mask: Binary mask indicating foreground regions
        return_large_mask: If 0, return only small mask; if 1, return both small and large masks

    Returns:
        If return_large_mask=0: small_mask
        If return_large_mask=1: (small_mask, large_mask)
    """
    small_mask = np.zeros((feature_size, feature_size))

    # Keep generating until we get a non-zero mask
    while np.max(small_mask) == 0:
        # Generate two threshold masks
        threshold_mask_1 = generate_threshold_mask(image_shape, min_scale, max_scale)
        threshold_mask_2 = generate_threshold_mask(image_shape, min_scale, max_scale)

        # Randomly combine the two masks
        random_value = torch.rand(1).numpy()[0]
        if random_value > 2 / 3:
            # Union of two masks
            combined_mask = threshold_mask_1 + threshold_mask_2
            combined_mask = np.where(
                combined_mask > 0,
                np.ones_like(combined_mask),
                np.zeros_like(combined_mask)
            )
        elif random_value > 1 / 3:
            # Intersection of two masks
            combined_mask = threshold_mask_1 * threshold_mask_2
        else:
            # Use only first mask
            combined_mask = threshold_mask_1

        # Apply foreground mask
        combined_mask = torch.from_numpy(combined_mask)
        foreground_masked = combined_mask * foreground_mask

        # Downsample to feature size
        downsample_ratio_y = int(image_shape[1] / feature_size)
        downsample_ratio_x = int(image_shape[2] / feature_size)

        large_mask = foreground_masked
        small_mask = torch.nn.functional.max_pool2d(
            foreground_masked.unsqueeze(0).unsqueeze(0),
            (downsample_ratio_y, downsample_ratio_x)
        ).float()
        small_mask = small_mask.numpy()[0, 0]

    if return_large_mask != 0:
        large_mask_np = large_mask.numpy()
        return small_mask, large_mask_np
    else:
        return small_mask


def linear_interpolate(x, y, weight):
    """Linear interpolation between x and y.

    Args:
        x: Start value
        y: End value
        weight: Interpolation weight (0 to 1)

    Returns:
        Interpolated value
    """
    return (y - x) * weight + x


def generate_perlin_noise_2d(shape, resolution, fade=lambda t: 6 * t ** 5 - 15 * t ** 4 + 10 * t ** 3):
    """Generate 2D Perlin noise.

    Args:
        shape: Output shape (height, width)
        resolution: Resolution of the noise grid (res_y, res_x)
        fade: Fade function for smooth interpolation

    Returns:
        2D Perlin noise array
    """
    delta = (resolution[0] / shape[0], resolution[1] / shape[1])
    grid_divisions = (shape[0] // resolution[0], shape[1] // resolution[1])
    grid = np.mgrid[0:resolution[0]:delta[0], 0:resolution[1]:delta[1]].transpose(1, 2, 0) % 1

    # Generate random gradient angles
    angles = 2 * math.pi * np.random.rand(resolution[0] + 1, resolution[1] + 1)
    gradients = np.stack((np.cos(angles), np.sin(angles)), axis=-1)

    # Tile gradients to match shape
    def tile_gradients(slice_y, slice_x):
        return np.repeat(
            np.repeat(
                gradients[slice_y[0]:slice_y[1], slice_x[0]:slice_x[1]],
                grid_divisions[0],
                axis=0
            ),
            grid_divisions[1],
            axis=1
        )

    # Compute dot products
    def compute_dot_product(gradient, shift):
        return (
            np.stack(
                (grid[:shape[0], :shape[1], 0] + shift[0],
                 grid[:shape[0], :shape[1], 1] + shift[1]),
                axis=-1
            ) * gradient[:shape[0], :shape[1]]
        ).sum(axis=-1)

    # Compute corner dot products
    corner_00 = compute_dot_product(tile_gradients([0, -1], [0, -1]), [0, 0])
    corner_10 = compute_dot_product(tile_gradients([1, None], [0, -1]), [-1, 0])
    corner_01 = compute_dot_product(tile_gradients([0, -1], [1, None]), [0, -1])
    corner_11 = compute_dot_product(tile_gradients([1, None], [1, None]), [-1, -1])

    # Apply fade function and interpolate
    fade_weights = fade(grid[:shape[0], :shape[1]])

    return math.sqrt(2) * linear_interpolate(
        linear_interpolate(corner_00, corner_10, fade_weights[..., 0]),
        linear_interpolate(corner_01, corner_11, fade_weights[..., 0]),
        fade_weights[..., 1]
    )
