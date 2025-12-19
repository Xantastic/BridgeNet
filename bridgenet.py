from loss import FocalLoss
from collections import OrderedDict
from torch.utils.tensorboard import SummaryWriter
from model import Discriminator, Projection, PatchMaker

import numpy as np
import torch.nn.functional as F
from torch import nn

import logging
import os
import math
import torch
import tqdm
import common
import metrics
import utils
import shutil
import cv2
import fnmatch
import glob

LOGGER = logging.getLogger(__name__)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class TensorBoardWrapper:
    """Wrapper for TensorBoard logging functionality."""

    def __init__(self, log_dir):
        self.global_iteration = 0
        self.logger = SummaryWriter(log_dir=log_dir)

    def step(self):
        """Increment the global iteration counter."""
        self.global_iteration += 1


class BridgeNet(torch.nn.Module):
    """
    BridgeNet model for anomaly detection using RGB and depth data.
    Combines features from both modalities for robust anomaly detection.
    """

    def __init__(self, device):
        super(BridgeNet, self).__init__()
        self.device = device

    def load(
        self,
        backbone_network,
        backbone_depth_network,
        layers_to_extract_from,
        device,
        input_shape,
        pretrain_embed_dimension,
        target_embed_dimension,
        patchsize=3,
        patchstride=1,
        meta_epochs=640,
        eval_epochs=1,
        dsc_layers=2,
        dsc_hidden=1024,
        dsc_margin=0.5,
        train_backbone=False,
        pre_proj=1,
        noise=0.015,
        lr=0.0001,
        svd=0,
        step=20,
        limit=392,
        down_scale=16,
        **kwargs,
    ):
        """Initialize the BridgeNet model with specified parameters."""

        # Network architectures
        self.backbone_network = backbone_network.to(device)
        self.backbone_depth_network = backbone_depth_network.to(device)
        self.layers_to_extract_from = layers_to_extract_from
        self.input_shape = input_shape
        self.device = device

        # Feature aggregation modules
        self.forward_modules = torch.nn.ModuleDict({})

        # Depth feature aggregator
        feature_aggregator_depth = common.NetworkFeatureAggregatorDepth(
            self.backbone_network, self.layers_to_extract_from, self.device, train_backbone
        )
        feature_dimensions_depth = feature_aggregator_depth.feature_dimensions(input_shape)
        self.forward_modules["feature_aggregator_depth"] = feature_aggregator_depth

        # RGB feature aggregator
        feature_aggregator = common.NetworkFeatureAggregator(
            self.backbone_depth_network, self.layers_to_extract_from, self.device, train_backbone
        )
        feature_dimensions = feature_aggregator.feature_dimensions(input_shape)
        self.forward_modules["feature_aggregator"] = feature_aggregator

        # Preprocessing modules
        preprocessing = common.Preprocessing(feature_dimensions, pretrain_embed_dimension)
        self.forward_modules["preprocessing"] = preprocessing

        preprocessing_depth = common.Preprocessing(feature_dimensions_depth, pretrain_embed_dimension)
        self.forward_modules["preprocessing_depth"] = preprocessing_depth

        # Target embedding dimension
        self.target_embed_dimension = target_embed_dimension

        # Feature aggregators
        preadapt_aggregator = common.Aggregator(target_dim=target_embed_dimension)
        preadapt_aggregator.to(self.device)
        self.forward_modules["preadapt_aggregator"] = preadapt_aggregator

        preadapt_aggregator_depth = common.Aggregator(target_dim=target_embed_dimension)
        preadapt_aggregator_depth.to(self.device)
        self.forward_modules["preadapt_aggregator_depth"] = preadapt_aggregator_depth

        # Training parameters
        self.meta_epochs = meta_epochs
        self.learning_rate = lr
        self.train_backbone = train_backbone

        # Backbone optimizer
        if self.train_backbone:
            self.backbone_optimizer = torch.optim.AdamW(
                self.forward_modules["feature_aggregator"].backbone.parameters(),
                lr
            )
            self.backbone_depth_optimizer = torch.optim.AdamW(
                self.forward_modules["feature_aggregator_depth"].backbone.parameters(),
                lr
            )

        # Pre-projection settings
        self.pre_proj = pre_proj
        if self.pre_proj <= 0:
            raise ValueError("pre_proj must be greater than 0")

        if self.pre_proj > 0:
            self.pre_projection = Projection(
                self.target_embed_dimension * 2,
                self.target_embed_dimension * 2,
                pre_proj
            )
            self.pre_projection.to(self.device)
            self.projection_optimizer = torch.optim.Adam(
                self.pre_projection.parameters(),
                lr,
                weight_decay=1e-5
            )

        # Evaluation parameters
        self.eval_epochs = eval_epochs
        self.dsc_layers = dsc_layers
        self.dsc_hidden = dsc_hidden

        # Discriminator
        self.discriminator = Discriminator(
            self.target_embed_dimension * 2,
            n_layers=dsc_layers,
            hidden=dsc_hidden * 2
        )
        self.discriminator.to(self.device)
        self.discriminator_optimizer = torch.optim.AdamW(
            self.discriminator.parameters(),
            lr * 2
        )
        self.dsc_margin = dsc_margin

        # Training parameters
        self.positive_center = torch.tensor(0)
        self.negative_center = torch.tensor(0)
        self.noise_level = noise
        self.svd = svd
        self.step = step
        self.sample_limit = limit
        self.distribution = 0
        self.focal_loss = FocalLoss()

        # Patch processing
        self.patch_maker = PatchMaker(patchsize, stride=patchstride)
        self.anomaly_segmentor = common.RescaleSegmentor(
            device=self.device,
            target_size=input_shape[-2:]
        )

        # Model directories
        self.model_dir = ""
        self.dataset_name = ""
        self.logger = None

        # Pixel unshuffle for downsampling
        self.pixel_unshuffle = nn.PixelUnshuffle(down_scale // 2)

    def set_model_dir(self, model_dir, dataset_name):
        """Set up model directories and logging."""
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)
        self.checkpoint_dir = os.path.join(self.model_dir, dataset_name)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.tensorboard_dir = os.path.join(self.checkpoint_dir, "tb")
        os.makedirs(self.tensorboard_dir, exist_ok=True)
        self.logger = TensorBoardWrapper(self.tensorboard_dir)

    def get_rgb_depth_noise(self, scale_factor, rgb_image, depth_image):
        """Generate noise for RGB and depth images."""
        random_value = np.random.rand()

        if random_value > 0.66:
            # Add noise to both RGB and depth
            rgb_noise = torch.normal(0, self.noise_level * scale_factor, rgb_image.shape).to(self.device)
            depth_noise = torch.normal(0, self.noise_level * scale_factor, depth_image.shape).to(self.device)
        elif random_value > 0.33:
            # Add noise only to RGB
            rgb_noise = torch.normal(0, self.noise_level * scale_factor, rgb_image.shape).to(self.device)
            depth_noise = 0
        else:
            # Add noise only to depth
            rgb_noise = 0
            depth_noise = torch.normal(0, self.noise_level * scale_factor, depth_image.shape).to(self.device)

        return rgb_noise, depth_noise

    def embed_features(
        self,
        images,
        depths=None,
        detach=True,
        provide_patch_shapes=False,
        evaluation=False
    ):
        """Extract and embed features from RGB and depth images."""

        # Extract RGB features
        if not evaluation and self.train_backbone:
            self.forward_modules["feature_aggregator"].train()
            rgb_features = self.forward_modules["feature_aggregator"](images, eval=evaluation)
        else:
            self.forward_modules["feature_aggregator"].eval()
            with torch.no_grad():
                rgb_features = self.forward_modules["feature_aggregator"](images)

        # Extract depth features
        depth_features = None
        if depths is not None:
            self.forward_modules["feature_aggregator_depth"].eval()
            with torch.no_grad():
                depth_features = self.forward_modules["feature_aggregator_depth"](depths)

        # Process RGB features
        rgb_features = [rgb_features[layer] for layer in self.layers_to_extract_from]

        # Process depth features
        if depth_features is not None:
            depth_features = [depth_features[layer] for layer in self.layers_to_extract_from]

        # Reshape transformer features if needed
        for i, feature in enumerate(rgb_features):
            if len(feature.shape) == 3:
                batch_size, sequence_length, channels = feature.shape
                rgb_features[i] = feature.reshape(
                    batch_size,
                    int(math.sqrt(sequence_length)),
                    int(math.sqrt(sequence_length)),
                    channels
                ).permute(0, 3, 1, 2)

        # Reshape transformer depth features if needed
        if depth_features is not None:
            for i, depth_feature in enumerate(depth_features):
                if len(depth_feature.shape) == 3:
                    batch_size, sequence_length, channels = depth_feature.shape
                    depth_features[i] = depth_feature.reshape(
                        batch_size,
                        int(math.sqrt(sequence_length)),
                        int(math.sqrt(sequence_length)),
                        channels
                    ).permute(0, 3, 1, 2)

        # Create patches from RGB features
        rgb_patches = [self.patch_maker.patchify(x, return_spatial_info=True) for x in rgb_features]
        rgb_patch_shapes = [x[1] for x in rgb_patches]
        rgb_patch_features = [x[0] for x in rgb_patches]

        # Create patches from depth features
        depth_patches = None
        depth_patch_shapes = None
        depth_patch_features = None

        if depth_features is not None:
            depth_patches = [self.patch_maker.patchify(x, return_spatial_info=True) for x in depth_features]
            depth_patch_shapes = [x[1] for x in depth_patches]
            depth_patch_features = [x[0] for x in depth_patches]

        # Reference patch dimensions
        reference_rgb_patches = rgb_patch_shapes[0]
        reference_depth_patches = depth_patch_shapes[0] if depth_patch_shapes is not None else None

        # Process multi-scale features
        for i in range(1, len(rgb_patch_features)):
            current_rgb_features = rgb_patch_features[i]
            current_depth_features = depth_patch_features[i] if depth_patch_features is not None else None
            current_patch_dims = rgb_patch_shapes[i]

            # Reshape RGB features
            current_rgb_features = current_rgb_features.reshape(
                current_rgb_features.shape[0],
                current_patch_dims[0],
                current_patch_dims[1],
                *current_rgb_features.shape[2:]
            )

            # Reshape depth features
            if current_depth_features is not None:
                current_depth_features = current_depth_features.reshape(
                    current_depth_features.shape[0],
                    current_patch_dims[0],
                    current_patch_dims[1],
                    *current_depth_features.shape[2:]
                )

            # Permute dimensions for interpolation
            current_rgb_features = current_rgb_features.permute(0, -3, -2, -1, 1, 2)
            if current_depth_features is not None:
                current_depth_features = current_depth_features.permute(0, -3, -2, -1, 1, 2)

            # Store original shape
            original_shape = current_rgb_features.shape

            # Reshape for interpolation
            current_rgb_features = current_rgb_features.reshape(-1, *current_rgb_features.shape[-2:])
            if current_depth_features is not None:
                current_depth_features = current_depth_features.reshape(-1, *current_depth_features.shape[-2:])

            # Interpolate to reference size
            current_rgb_features = F.interpolate(
                current_rgb_features.unsqueeze(1),
                size=(reference_rgb_patches[0], reference_rgb_patches[1]),
                mode="bilinear",
                align_corners=False,
            )

            if current_depth_features is not None:
                current_depth_features = F.interpolate(
                    current_depth_features.unsqueeze(1),
                    size=(reference_rgb_patches[0], reference_rgb_patches[1]),
                    mode="bilinear",
                    align_corners=False,
                )

            # Squeeze and reshape
            current_rgb_features = current_rgb_features.squeeze(1)
            if current_depth_features is not None:
                current_depth_features = current_depth_features.squeeze(1)

            current_rgb_features = current_rgb_features.reshape(
                *original_shape[:-2],
                reference_rgb_patches[0],
                reference_rgb_patches[1]
            )

            if current_depth_features is not None:
                current_depth_features = current_depth_features.reshape(
                    *original_shape[:-2],
                    reference_rgb_patches[0],
                    reference_rgb_patches[1]
                )

            # Permute back and finalize
            current_rgb_features = current_rgb_features.permute(0, -2, -1, 1, 2, 3)
            if current_depth_features is not None:
                current_depth_features = current_depth_features.permute(0, -2, -1, 1, 2, 3)

            current_rgb_features = current_rgb_features.reshape(
                len(current_rgb_features),
                -1,
                *current_rgb_features.shape[-3:]
            )

            if current_depth_features is not None:
                current_depth_features = current_depth_features.reshape(
                    len(current_depth_features),
                    -1,
                    *current_depth_features.shape[-3:]
                )

            # Update feature lists
            rgb_patch_features[i] = current_rgb_features
            if current_depth_features is not None:
                depth_patch_features[i] = current_depth_features

        # Process patches through preprocessing and aggregation
        rgb_patch_features = [x.reshape(-1, *x.shape[-3:]) for x in rgb_patch_features]
        rgb_patch_features = self.forward_modules["preprocessing"](rgb_patch_features)
        rgb_patch_features = self.forward_modules["preadapt_aggregator"](rgb_patch_features)

        if depth_patch_features is not None:
            depth_patch_features = [x.reshape(-1, *x.shape[-3:]) for x in depth_patch_features]
            depth_patch_features = self.forward_modules["preprocessing_depth"](depth_patch_features)
            depth_patch_features = self.forward_modules["preadapt_aggregator_depth"](depth_patch_features)

        return rgb_patch_features, rgb_patch_shapes, depth_patch_features

    def trainer(self, training_data, validation_data, dataset_name):
        """Train the BridgeNet model."""
        model_state_dict = {}
        checkpoint_paths = glob.glob(self.checkpoint_dir + '/ckpt_best*')
        checkpoint_save_path = os.path.join(self.checkpoint_dir, "ckpt.pth")

        if len(checkpoint_paths) != 0:
            LOGGER.info("Start testing, checkpoint file found!")
            return 0., 0., 0., 0., 0., -1.

        def update_model_state_dict():
            """Update the model state dictionary."""
            model_state_dict["discriminator"] = OrderedDict({
                key: value.detach().cpu()
                for key, value in self.discriminator.state_dict().items()
            })
            if self.pre_proj > 0:
                model_state_dict["pre_projection"] = OrderedDict({
                    key: value.detach().cpu()
                    for key, value in self.pre_projection.state_dict().items()
                })

        # Training progress bar
        progress_bar = tqdm.tqdm(range(self.meta_epochs), unit='epoch')
        progress_bar_string = ""
        best_performance_record = None

        for current_epoch in progress_bar:
            self.forward_modules.eval()

            # Train discriminator
            progress_bar_str, _, _ = self._train_discriminator(
                training_data,
                current_epoch,
                progress_bar,
                progress_bar_string
            )
            update_model_state_dict()

            # Evaluate model
            if (current_epoch + 1) % self.eval_epochs == 0:
                images, scores, segmentations, ground_truth_labels, ground_truth_masks = self.predict(validation_data)

                image_auroc, image_ap, pixel_auroc, pixel_ap, pixel_pro = self._evaluate(
                    images,
                    scores,
                    segmentations,
                    ground_truth_labels,
                    ground_truth_masks,
                    dataset_name
                )

                # Log metrics
                self.logger.logger.add_scalar("image_auroc", image_auroc, current_epoch)
                self.logger.logger.add_scalar("image_ap", image_ap, current_epoch)
                self.logger.logger.add_scalar("pixel_auroc", pixel_auroc, current_epoch)
                self.logger.logger.add_scalar("pixel_ap", pixel_ap, current_epoch)
                self.logger.logger.add_scalar("pixel_pro", pixel_pro, current_epoch)

                # Save visualization paths
                evaluation_save_path = './results/eval/' + dataset_name + '/'
                training_save_path = './results/training/' + dataset_name + '/'

                # Update best model
                if best_performance_record is None:
                    best_performance_record = [
                        image_auroc, image_ap, pixel_auroc, pixel_ap, pixel_pro, current_epoch
                    ]
                    checkpoint_best_path = os.path.join(
                        self.checkpoint_dir,
                        f"{image_auroc}_{pixel_auroc}_ckpt_best_{current_epoch}.pth"
                    )
                    torch.save(model_state_dict, checkpoint_best_path)
                    shutil.rmtree(evaluation_save_path, ignore_errors=True)
                    shutil.copytree(training_save_path, evaluation_save_path)

                elif image_auroc + pixel_auroc > best_performance_record[0] + best_performance_record[2]:
                    best_performance_record = [
                        image_auroc, image_ap, pixel_auroc, pixel_ap, pixel_pro, current_epoch
                    ]
                    os.remove(checkpoint_best_path)
                    checkpoint_best_path = os.path.join(
                        self.checkpoint_dir,
                        f"{image_auroc}_{pixel_auroc}_ckpt_best_{current_epoch}.pth"
                    )
                    torch.save(model_state_dict, checkpoint_best_path)
                    shutil.rmtree(evaluation_save_path, ignore_errors=True)
                    shutil.copytree(training_save_path, evaluation_save_path)

                # Update progress bar description
                progress_bar_string = (
                    f" IAUC:{round(image_auroc * 100, 2)}({round(best_performance_record[0] * 100, 2)})"
                    f" IAP:{round(image_ap * 100, 2)}({round(best_performance_record[1] * 100, 2)})"
                    f" PAUC:{round(pixel_auroc * 100, 2)}({round(best_performance_record[2] * 100, 2)})"
                    f" PAP:{round(pixel_ap * 100, 2)}({round(best_performance_record[3] * 100, 2)})"
                    f" PRO:{round(pixel_pro * 100, 2)}({round(best_performance_record[4] * 100, 2)})"
                    f" E:{current_epoch}({best_performance_record[-1]})"
                )
                progress_bar_str += progress_bar_string
                progress_bar.set_description_str(progress_bar_str)

            # Save checkpoint
            torch.save(model_state_dict, checkpoint_save_path)

        return best_performance_record

    def _train_discriminator(self, training_data, current_epoch, progress_bar, progress_bar_string):
        """Train the discriminator component of BridgeNet."""
        self.forward_modules.eval()
        if self.pre_proj > 0:
            self.pre_projection.train()
        self.discriminator.train()

        # Loss tracking
        discriminator_losses = []
        sample_count = 0

        for iteration, data_batch in enumerate(training_data):
            self.discriminator_optimizer.zero_grad()
            if self.pre_proj > 0:
                self.projection_optimizer.zero_grad()

            # Extract data
            augmented_rgb_image = data_batch["aug_image"].to(torch.float).to(self.device)
            original_rgb_image = data_batch["image"].to(torch.float).to(self.device)
            augmented_depth_image = data_batch["aug_depth"].to(torch.float).to(self.device)
            original_depth_image = data_batch["depth"].to(torch.float).to(self.device)

            # Generate noisy features if pre-projection is enabled
            if self.pre_proj > 0:
                # First noise level
                rgb_noise_level_low, depth_noise_level_low = self.get_rgb_depth_noise(
                    6, original_rgb_image, original_depth_image
                )
                noisy_rgb_image_low = original_rgb_image + rgb_noise_level_low
                noisy_depth_image_low = original_depth_image + depth_noise_level_low

                noisy_rgb_features_low, _, noisy_depth_features_low = self.embed_features(
                    noisy_rgb_image_low, depths=noisy_depth_image_low, evaluation=False
                )
                noisy_features_low = torch.cat((noisy_rgb_features_low, noisy_depth_features_low), dim=1)
                noisy_features_low = self.pre_projection(noisy_features_low)

                # Fake features (augmented data)
                fake_rgb_features, _, fake_depth_features = self.embed_features(
                    augmented_rgb_image, depths=augmented_depth_image, evaluation=False
                )
                fake_features = torch.cat((fake_rgb_features, fake_depth_features), dim=1)
                fake_features = self.pre_projection(fake_features)

                # True features (original data)
                true_rgb_features, _, true_depth_features = self.embed_features(
                    original_rgb_image, depths=original_depth_image, evaluation=False
                )

                # Second noise level
                rgb_noise_level_medium, depth_noise_level_medium = self.get_rgb_depth_noise(
                    2, true_rgb_features, true_depth_features
                )
                noisy_rgb_features_medium = true_rgb_features + rgb_noise_level_medium
                noisy_depth_features_medium = true_depth_features + depth_noise_level_medium
                noisy_features_medium = torch.cat((noisy_rgb_features_medium, noisy_depth_features_medium), dim=1)

                true_features = torch.cat((true_rgb_features, true_depth_features), dim=1)
                noisy_features_medium = self.pre_projection(noisy_features_medium)
                true_features = self.pre_projection(true_features)

                # Third noise level
                noise_level_large = torch.normal(0, self.noise_level, true_features.shape).to(self.device)
                noisy_features_large = true_features + noise_level_large
            else:
                raise ValueError("pre_proj must be greater than 0. Please set --pre_proj parameter to a positive value.")

            # Ground truth masks
            segmentation_mask = data_batch["mask_s"].reshape(-1, 1).to(self.device)
            segmentation_mask_low = data_batch["mask_s_0"].reshape(-1, 1).to(self.device)
            segmentation_mask_medium = data_batch["mask_s_1"].reshape(-1, 1).to(self.device)

            # Discriminator losses for different noise levels
            if self.pre_proj > 0:
                # First level (low)
                scores_low = self.discriminator(torch.cat([true_features, noisy_features_low]))
                true_scores_low = scores_low[:len(true_features)]
                noisy_scores_low = scores_low[len(true_features):]
                true_loss_low = torch.nn.BCELoss()(true_scores_low, torch.zeros_like(true_scores_low))
                noisy_loss_low = torch.nn.BCELoss()(noisy_scores_low, torch.ones_like(noisy_scores_low))
                bce_loss_low = true_loss_low + noisy_loss_low

                # Second level (medium)
                scores_medium = self.discriminator(torch.cat([true_features, noisy_features_medium]))
                true_scores_medium = scores_medium[:len(true_features)]
                noisy_scores_medium = scores_medium[len(true_features):]
                true_loss_medium = torch.nn.BCELoss()(true_scores_medium, torch.zeros_like(true_scores_medium))
                noisy_loss_medium = torch.nn.BCELoss()(noisy_scores_medium, torch.ones_like(noisy_scores_medium))
                bce_loss_medium = true_loss_medium + noisy_loss_medium

                # Third level (large)
                scores_large = self.discriminator(torch.cat([true_features, noisy_features_large]))
                true_scores_large = scores_large[:len(true_features)]
                noisy_scores_large = scores_large[len(true_features):]
                true_loss_large = torch.nn.BCELoss()(true_scores_large, torch.zeros_like(true_scores_large))
                noisy_loss_large = torch.nn.BCELoss()(noisy_scores_large, torch.ones_like(noisy_scores_large))
                bce_loss_large = true_loss_large + noisy_loss_large

                # Total BCE loss
                total_bce_loss = bce_loss_low + bce_loss_medium + bce_loss_large

                # Focal loss for fake features
                fake_scores = self.discriminator(fake_features)
                fake_scores_prob = fake_scores
                mask = segmentation_mask
                output = torch.cat([1 - fake_scores_prob, fake_scores_prob], dim=1)
                focal_loss = self.focal_loss(output, mask)

                # Total loss
                total_loss = total_bce_loss + 1.0 * focal_loss

                # Backpropagation
                total_loss.backward()
                if self.pre_proj > 0:
                    self.projection_optimizer.step()
                if self.train_backbone:
                    self.backbone_optimizer.step()
                    self.backbone_depth_optimizer.step()
                self.discriminator_optimizer.step()

                # Log loss
                self.logger.logger.add_scalar("discriminator_loss", total_loss, self.logger.global_iteration)
                self.logger.step()

                discriminator_losses.append(total_loss.detach().cpu().item())
                average_loss = np.mean(discriminator_losses)
                sample_count += original_rgb_image.shape[0]

                # Update progress bar
                progress_bar_description = f"epoch:{current_epoch} loss:{average_loss:.2e}"
                progress_bar_description += f" sample:{sample_count}"
                final_progress_bar_string = progress_bar_description
                final_progress_bar_string += progress_bar_string
                progress_bar.set_description_str(final_progress_bar_string)

                # Check sample limit
                if sample_count > self.sample_limit:
                    break

        return final_progress_bar_string, 0, 0

    def tester(self, test_data, dataset_name):
        """Test the BridgeNet model."""
        # Use os.listdir and fnmatch instead of glob.glob due to potential issues
        # Match any file containing "ckpt_best" anywhere in the filename
        checkpoint_paths = []
        if os.path.isdir(self.checkpoint_dir):
            for filename in os.listdir(self.checkpoint_dir):
                if fnmatch.fnmatch(filename, '*ckpt_best*'):
                    checkpoint_paths.append(os.path.join(self.checkpoint_dir, filename))

        if len(checkpoint_paths) != 0:
            model_state_dict = torch.load(checkpoint_paths[0], map_location=self.device)
            if 'discriminator' in model_state_dict:
                self.discriminator.load_state_dict(model_state_dict['discriminator'])
                if "pre_projection" in model_state_dict:
                    self.pre_projection.load_state_dict(model_state_dict["pre_projection"])
            else:
                self.load_state_dict(model_state_dict, strict=False)

            images, scores, segmentations, ground_truth_labels, ground_truth_masks = self.predict(test_data)
            image_auroc, image_ap, pixel_auroc, pixel_ap, pixel_pro = self._evaluate(
                images,
                scores,
                segmentations,
                ground_truth_labels,
                ground_truth_masks,
                dataset_name,
                path='eval'
            )
            epoch = int(checkpoint_paths[0].split('_')[-1].split('.')[0])
        else:
            image_auroc, image_ap, pixel_auroc, pixel_ap, pixel_pro, epoch = 0., 0., 0., 0., 0., -1.
            LOGGER.info("No checkpoint file found!")

        return image_auroc, image_ap, pixel_auroc, pixel_ap, pixel_pro, epoch

    def _evaluate(
        self,
        images,
        scores,
        segmentations,
        ground_truth_labels,
        ground_truth_masks,
        dataset_name,
        path='training'
    ):
        """Evaluate model performance."""
        scores = np.squeeze(np.array(scores))
        min_score = min(scores)
        max_score = max(scores)
        normalized_scores = (scores - min_score) / (max_score - min_score + 1e-10)

        # Compute image-level metrics
        image_scores = metrics.compute_imagewise_retrieval_metrics(
            normalized_scores,
            ground_truth_labels,
            path
        )
        image_auroc = image_scores["auroc"]
        image_ap = image_scores["ap"]

        # Compute pixel-level metrics
        if len(ground_truth_masks) > 0:
            segmentations = np.array(segmentations)
            min_seg_score = np.min(segmentations)
            max_seg_score = np.max(segmentations)
            normalized_segmentations = (segmentations - min_seg_score) / (max_seg_score - min_seg_score + 1e-10)

            pixel_scores = metrics.compute_pixelwise_retrieval_metrics(
                normalized_segmentations,
                ground_truth_masks,
                path
            )
            pixel_auroc = pixel_scores["auroc"]
            pixel_ap = pixel_scores["ap"]

            if path == 'eval':
                try:
                    pixel_pro = metrics.compute_pro(
                        np.squeeze(np.array(ground_truth_masks)),
                        normalized_segmentations
                    )
                except:
                    pixel_pro = 0.
            else:
                pixel_pro = 0.
        else:
            pixel_auroc = -1.
            pixel_ap = -1.
            pixel_pro = -1.
            return image_auroc, image_ap, pixel_auroc, pixel_ap, pixel_pro

        # Generate visualization
        self._save_visualization(
            images,
            ground_truth_masks,
            normalized_segmentations,
            dataset_name,
            path
        )

        return image_auroc, image_ap, pixel_auroc, pixel_ap, pixel_pro

    def _save_visualization(
        self,
        images,
        ground_truth_masks,
        normalized_segmentations,
        dataset_name,
        path
    ):
        """Save visualization results."""
        defect_images = np.array(images)
        target_masks = np.array(ground_truth_masks)

        for i in range(len(defect_images)):
            defect_image = utils.torch_format_2_numpy_img(defect_images[i])
            target_mask = utils.torch_format_2_numpy_img(target_masks[i])

            # Create colored segmentation map
            mask = cv2.cvtColor(
                cv2.resize(
                    normalized_segmentations[i],
                    (defect_image.shape[1], defect_image.shape[0])
                ),
                cv2.COLOR_GRAY2BGR
            )
            mask = (mask * 255).astype('uint8')
            mask = cv2.applyColorMap(mask, cv2.COLORMAP_JET)

            # Combine images for visualization
            combined_image = np.hstack([defect_image, target_mask, mask])
            combined_image = cv2.resize(combined_image, (256 * 3, 256))

            # Save visualization
            save_directory = './results/' + path + '/' + dataset_name + '/'
            utils.del_remake_dir(save_directory, del_flag=False)
            cv2.imwrite(save_directory + str(i + 1).zfill(3) + '.png', combined_image)

    def predict(self, test_dataloader):
        """Generate predictions for the test dataset."""
        self.forward_modules.eval()

        image_paths = []
        images = []
        depths = []
        scores = []
        masks = []
        ground_truth_labels = []
        ground_truth_masks = []

        with tqdm.tqdm(test_dataloader, desc="Inferring...", leave=False, unit='batch') as data_iterator:
            for data in data_iterator:
                if isinstance(data, dict):
                    ground_truth_labels.extend(data["is_anomaly"].numpy().tolist())
                    if data.get("mask_gt", None) is not None:
                        ground_truth_masks.extend(data["mask_gt"].numpy().tolist())
                    image = data["image"]
                    depth = data["depth"]
                    images.extend(image.numpy().tolist())
                    depths.extend(image.numpy().tolist())
                    image_paths.extend(data["image_path"])

                _scores, _masks = self._predict(image, depth)
                for score, mask in zip(_scores, _masks):
                    scores.append(score)
                    masks.append(mask)

        return images, scores, masks, ground_truth_labels, ground_truth_masks

    def _predict(self, image, depth):
        """Generate predictions for a batch of images."""
        image = image.to(torch.float).to(self.device)
        depth = depth.to(torch.float).to(self.device)
        self.forward_modules.eval()

        if self.pre_proj > 0:
            self.pre_projection.eval()
        self.discriminator.eval()

        with torch.no_grad():
            # Extract features
            patch_features, patch_shapes, depth_features = self.embed_features(
                image,
                depths=depth,
                provide_patch_shapes=True,
                evaluation=True
            )

            # Combine RGB and depth features
            patch_features = torch.cat((patch_features, depth_features), dim=1)

            # Apply pre-projection if enabled
            if self.pre_proj > 0:
                patch_features = self.pre_projection(patch_features)
                patch_features = patch_features[0] if len(patch_features) == 2 else patch_features

            # Generate scores
            patch_scores = image_scores = self.discriminator(patch_features)

            # Convert patch scores to segmentation masks
            patch_scores = self.patch_maker.unpatch_scores(patch_scores, batchsize=image.shape[0])
            scales = patch_shapes[0]
            patch_scores = patch_scores.reshape(image.shape[0], scales[0], scales[1])
            masks = self.anomaly_segmentor.convert_to_segmentation(patch_scores)

            # Convert to image-level scores
            image_scores = self.patch_maker.unpatch_scores(image_scores, batchsize=image.shape[0])
            image_scores = self.patch_maker.score(image_scores)
            if isinstance(image_scores, torch.Tensor):
                image_scores = image_scores.cpu().numpy()

        return list(image_scores), list(masks)
