"""Perlin noise generation for creating synthetic anomaly masks."""

import math

import numpy as np
import torch
import imgaug.augmenters as iaa


def generate_perlin_threshold_mask(img_shape, min_scale=0, max_scale=4):
    """Generate a binary mask from Perlin noise.

    Args:
        img_shape: Shape of output mask [channels, height, width].
        min_scale: Minimum power of 2 for Perlin scale.
        max_scale: Maximum power of 2 for Perlin scale.

    Returns:
        Binary mask with values {0, 1}.
    """
    perlin_scale_x = 2 ** np.random.randint(min_scale, max_scale)
    perlin_scale_y = 2 ** np.random.randint(min_scale, max_scale)

    perlin_noise = generate_perlin_noise_2d(
        (img_shape[1], img_shape[2]), (perlin_scale_x, perlin_scale_y)
    )

    # Apply random rotation
    perlin_noise = iaa.Sequential([iaa.Affine(rotate=(-90, 90))])(image=perlin_noise)

    # Apply threshold
    threshold = 0.5
    binary_mask = np.where(perlin_noise > threshold, 1.0, 0.0)

    return binary_mask


def generate_perlin_mask(img_shape, feature_size, min_scale, max_scale, foreground_mask, return_large=False):
    """Generate Perlin noise anomaly mask.

    Args:
        img_shape: Shape of input image [channels, height, width].
        feature_size: Size of downsampled mask.
        min_scale: Minimum power of 2 for Perlin scales.
        max_scale: Maximum power of 2 for Perlin scales.
        foreground_mask: Foreground mask to constrain anomaly region.
        return_large: If True, also return high-resolution mask.

    Returns:
        Downsampled mask, and optionally high-resolution mask if return_large=True.
    """
    mask = np.zeros((feature_size, feature_size))

    # Retry until non-empty mask is generated
    while np.max(mask) == 0:
        perlin_mask_1 = generate_perlin_threshold_mask(img_shape, min_scale, max_scale)
        perlin_mask_2 = generate_perlin_threshold_mask(img_shape, min_scale, max_scale)

        # Randomly combine masks
        combine_type = torch.rand(1).item()

        if combine_type > 2 / 3:
            # Addition
            combined_mask = perlin_mask_1 + perlin_mask_2
            combined_mask = np.where(combined_mask > 0, 1.0, 0.0)
        elif combine_type > 1 / 3:
            # Multiplication
            combined_mask = perlin_mask_1 * perlin_mask_2
        else:
            # Use first mask only
            combined_mask = perlin_mask_1

        # Apply foreground constraint
        combined_mask_tensor = torch.from_numpy(combined_mask)
        masked_perlin = combined_mask_tensor * foreground_mask

        # Downsample using max pooling
        downsample_y = img_shape[1] // feature_size
        downsample_x = img_shape[2] // feature_size

        mask_tensor = torch.nn.functional.max_pool2d(
            masked_perlin.unsqueeze(0).unsqueeze(0),
            (downsample_y, downsample_x)
        ).float()
        mask = mask_tensor.numpy()[0, 0]

        if return_large:
            high_res_mask = masked_perlin.numpy()

    if return_large:
        return mask, high_res_mask
    else:
        return mask


def lerp(x, y, weight):
    """Linear interpolation between x and y with weight."""
    return (y - x) * weight + x


def fade(t):
    """Fade function for Perlin noise: 6t^5 - 15t^4 + 10t^3."""
    return 6 * t ** 5 - 15 * t ** 4 + 10 * t ** 3


def generate_perlin_noise_2d(shape, resolution):
    """Generate 2D Perlin noise.

    Args:
        shape: Output shape (height, width).
        resolution: Resolution of Perlin noise (res_x, res_y).

    Returns:
        2D Perlin noise array.
    """
    delta = (resolution[0] / shape[0], resolution[1] / shape[1])
    tile_size = (shape[0] // resolution[0], shape[1] // resolution[1])

    # Generate grid
    grid = np.mgrid[0:resolution[0]:delta[0], 0:resolution[1]:delta[1]].transpose(1, 2, 0) % 1

    # Generate gradients
    angles = 2 * math.pi * np.random.rand(resolution[0] + 1, resolution[1] + 1)
    gradients = np.stack((np.cos(angles), np.sin(angles)), axis=-1)

    # Repeat gradients to match tile size
    gradient_tiles = np.repeat(
        np.repeat(gradients, tile_size[0], axis=0), tile_size[1], axis=1
    )

    # Helper function to tile gradients
    def tile_gradient(slice_x, slice_y):
        """Extract and tile gradient region."""
        region = gradients[slice_x[0]:slice_x[1], slice_y[0]:slice_y[1]]
        return np.repeat(np.repeat(region, tile_size[0], axis=0), tile_size[1], axis=1)

    # Helper function for dot product
    def dot_product(grad, shift):
        """Compute dot product of gradient and shifted grid."""
        shifted_x = grid[:shape[0], :shape[1], 0] + shift[0]
        shifted_y = grid[:shape[0], :shape[1], 1] + shift[1]
        shifted = np.stack((shifted_x, shifted_y), axis=-1)
        return (shifted * grad[:shape[0], :shape[1]]).sum(axis=-1)

    # Compute noise components
    n00 = dot_product(tile_gradient([0, -1], [0, -1]), [0, 0])
    n10 = dot_product(tile_gradient([1, None], [0, -1]), [-1, 0])
    n01 = dot_product(tile_gradient([0, -1], [1, None]), [0, -1])
    n11 = dot_product(tile_gradient([1, None], [1, None]), [-1, -1])

    # Smooth interpolation
    t = fade(grid[:shape[0], :shape[1]])

    # Combine components
    noise = math.sqrt(2) * lerp(
        lerp(n00, n10, t[..., 0]),
        lerp(n01, n11, t[..., 0]),
        t[..., 1]
    )

    return noise
