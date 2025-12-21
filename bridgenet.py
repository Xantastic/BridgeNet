from loss import FocalLoss
from collections import OrderedDict
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from model import Discriminator, Projection, PatchMaker

import numpy as np
import pandas as pd
import torch.nn.functional as F
from torch import nn

import logging
import os
import math
import torch
import tqdm
import common
import metrics
import cv2
import utils
import glob
import shutil

LOGGER = logging.getLogger(__name__)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class TBWrapper:
    def __init__(self, log_dir):
        self.g_iter = 0
        self.logger = SummaryWriter(log_dir=log_dir)

    def step(self):
        self.g_iter += 1


class BridgeNet(torch.nn.Module):
    def __init__(self, device):
        super(BridgeNet, self).__init__()
        self.device = device

    def load(
            self,
            backbone,
            backbone_depth,
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
            train_backbone=False,
            pre_proj=1,
            noise=0.015,
            lr=0.0001,
            step=20,
            limit=392,
            down_scale=16,
            **kwargs,
    ):

        self.backbone = backbone.to(device)
        self.backbone_depth = backbone_depth.to(device)
        self.layers_to_extract_from = layers_to_extract_from
        self.input_shape = input_shape
        self.device = device

        self.forward_modules = torch.nn.ModuleDict({})

        feature_aggregator_depth = common.NetworkFeatureAggregatorDepth(
            self.backbone, self.layers_to_extract_from, self.device, train_backbone
        )
        feature_dimensions_depth = feature_aggregator_depth.feature_dimensions(input_shape)
        self.forward_modules["feature_aggregator_depth"] = feature_aggregator_depth

        feature_aggregator = common.NetworkFeatureAggregator(
            self.backbone_depth, self.layers_to_extract_from, self.device, train_backbone
        )
        feature_dimensions = feature_aggregator.feature_dimensions(input_shape)
        self.forward_modules["feature_aggregator"] = feature_aggregator

        preprocessing = common.Preprocessing(feature_dimensions, pretrain_embed_dimension)
        self.forward_modules["preprocessing"] = preprocessing

        preprocessing_3D = common.Preprocessing3D(feature_dimensions_depth, pretrain_embed_dimension)
        self.forward_modules["preprocessing_3D"] = preprocessing_3D

        self.target_embed_dimension = target_embed_dimension
        preadapt_aggregator = common.Aggregator(target_dim=target_embed_dimension)
        preadapt_aggregator.to(self.device)
        self.forward_modules["preadapt_aggregator"] = preadapt_aggregator

        self.meta_epochs = meta_epochs
        self.lr = lr
        self.train_backbone = train_backbone
        if self.train_backbone:
            self.backbone_opt = torch.optim.AdamW(self.forward_modules["feature_aggregator"].backbone.parameters(), lr)

        self.pre_proj = pre_proj
        if self.pre_proj > 0:
            self.pre_projection = Projection(self.target_embed_dimension * 2, self.target_embed_dimension * 2, pre_proj)
            self.pre_projection.to(self.device)
            self.proj_opt = torch.optim.Adam(self.pre_projection.parameters(), lr, weight_decay=1e-5)


        self.eval_epochs = eval_epochs
        self.dsc_layers = dsc_layers
        self.dsc_hidden = dsc_hidden
        self.discriminator = Discriminator(self.target_embed_dimension * 2, n_layers=dsc_layers, hidden=dsc_hidden * 2)
        self.discriminator.to(self.device)
        self.dsc_opt = torch.optim.AdamW(self.discriminator.parameters(), lr=lr * 2)

        self.c = torch.tensor(0)
        self.c_ = torch.tensor(0)
        self.noise = noise
        self.step = step
        self.limit = limit
        self.focal_loss = FocalLoss()

        self.patch_maker = PatchMaker(patchsize, stride=patchstride)
        self.anomaly_segmentor = common.RescaleSegmentor(device=self.device, target_size=input_shape[-2:])
        self.model_dir = ""
        self.dataset_name = ""
        self.logger = None

        self.unshuffle = nn.PixelUnshuffle(down_scale//2)

    def set_model_dir(self, model_dir, dataset_name):
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)
        self.ckpt_dir = os.path.join(self.model_dir, dataset_name)
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.tb_dir = os.path.join(self.ckpt_dir, "tb")
        os.makedirs(self.tb_dir, exist_ok=True)
        self.logger = TBWrapper(self.tb_dir)

    def get_rgb_depth_noise(self, scale, image, depth):
        """Generate random noise for RGB and depth with probability-based selection."""
        random_value = np.random.rand()
        if random_value > 0.66:
            rgb_noise = torch.normal(0, self.noise * scale, image.shape).to(self.device)
            depth_noise = torch.normal(0, self.noise * scale, depth.shape).to(self.device)
        elif random_value > 0.33:
            rgb_noise = torch.normal(0, self.noise * scale, image.shape).to(self.device)
            depth_noise = 0
        else:
            rgb_noise = 0
            depth_noise = torch.normal(0, self.noise * scale, depth.shape).to(self.device)
        return rgb_noise, depth_noise



    def _embed(self, images, depths=None, detach=True, provide_patch_shapes=False, evaluation=False):
        """Extract feature embeddings from RGB images and depth maps.

        Args:
            images: RGB images tensor
            depths: Depth maps tensor (optional)
            detach: Whether to detach gradients (unused)
            provide_patch_shapes: Whether to return patch shape information
            evaluation: Whether in evaluation mode

        Returns:
            Tuple of (rgb_features, patch_shapes, depth_features)
        """
        if not evaluation and self.train_backbone:
            self.forward_modules["feature_aggregator"].train()
            rgb_features = self.forward_modules["feature_aggregator"](images, eval=evaluation)
        else:
            self.forward_modules["feature_aggregator"].eval()
            with torch.no_grad():
                rgb_features = self.forward_modules["feature_aggregator"](images)

        depth_features = None
        if depths is not None:
            self.forward_modules["feature_aggregator_depth"].eval()
            with torch.no_grad():
                depth_features = self.forward_modules["feature_aggregator_depth"](depths)



        # Extract features from specified layers
        rgb_features = [rgb_features[layer] for layer in self.layers_to_extract_from]
        depth_features = [depth_features[layer] for layer in self.layers_to_extract_from]

        # Reshape features if needed (for ViT-style outputs)
        for i, feature in enumerate(rgb_features):
            if len(feature.shape) == 3:
                batch_size, seq_len, channels = feature.shape
                spatial_size = int(math.sqrt(seq_len))
                rgb_features[i] = feature.reshape(batch_size, spatial_size, spatial_size, channels).permute(0, 3, 1, 2)

        for i, depth_feature in enumerate(depth_features):
            if len(depth_feature.shape) == 3:
                batch_size, seq_len, channels = depth_feature.shape
                spatial_size = int(math.sqrt(seq_len))
                depth_features[i] = depth_feature.reshape(batch_size, spatial_size, spatial_size, channels).permute(0, 3, 1, 2)

        # Convert features to patches
        rgb_patches_with_info = [self.patch_maker.patchify(x, return_spatial_info=True) for x in rgb_features]
        patch_shapes = [x[1] for x in rgb_patches_with_info]
        rgb_patch_features = [x[0] for x in rgb_patches_with_info]

        depth_patches_with_info = [self.patch_maker.patchify(x, return_spatial_info=True) for x in depth_features]
        depth_patch_shapes = [x[1] for x in depth_patches_with_info]
        depth_patch_features = [x[0] for x in depth_patches_with_info]

        reference_patch_size = patch_shapes[0]

        # Align all patches to the same spatial resolution
        for i in range(1, len(rgb_patch_features)):
            current_rgb_patches = rgb_patch_features[i]
            current_depth_patches = depth_patch_features[i]
            current_patch_dims = patch_shapes[i]

            # Reshape to spatial format
            current_rgb_patches = current_rgb_patches.reshape(
                current_rgb_patches.shape[0], current_patch_dims[0], current_patch_dims[1], *current_rgb_patches.shape[2:]
            )
            current_depth_patches = current_depth_patches.reshape(
                current_depth_patches.shape[0], current_patch_dims[0], current_patch_dims[1], *current_depth_patches.shape[2:]
            )

            # Permute for interpolation
            current_rgb_patches = current_rgb_patches.permute(0, -3, -2, -1, 1, 2)
            current_depth_patches = current_depth_patches.permute(0, -3, -2, -1, 1, 2)

            base_shape = current_rgb_patches.shape

            # Flatten for interpolation
            current_rgb_patches = current_rgb_patches.reshape(-1, *current_rgb_patches.shape[-2:])
            current_depth_patches = current_depth_patches.reshape(-1, *current_depth_patches.shape[-2:])

            # Interpolate to reference size
            current_rgb_patches = F.interpolate(
                current_rgb_patches.unsqueeze(1),
                size=(reference_patch_size[0], reference_patch_size[1]),
                mode="bilinear",
                align_corners=False,
            )
            current_depth_patches = F.interpolate(
                current_depth_patches.unsqueeze(1),
                size=(reference_patch_size[0], reference_patch_size[1]),
                mode="bilinear",
                align_corners=False,
            )

            current_rgb_patches = current_rgb_patches.squeeze(1)
            current_depth_patches = current_depth_patches.squeeze(1)

            # Reshape back
            current_rgb_patches = current_rgb_patches.reshape(
                *base_shape[:-2], reference_patch_size[0], reference_patch_size[1]
            )
            current_depth_patches = current_depth_patches.reshape(
                *base_shape[:-2], reference_patch_size[0], reference_patch_size[1]
            )

            # Permute back
            current_rgb_patches = current_rgb_patches.permute(0, -2, -1, 1, 2, 3)
            current_depth_patches = current_depth_patches.permute(0, -2, -1, 1, 2, 3)

            # Final reshape
            current_rgb_patches = current_rgb_patches.reshape(len(current_rgb_patches), -1, *current_rgb_patches.shape[-3:])
            current_depth_patches = current_depth_patches.reshape(len(current_depth_patches), -1, *current_depth_patches.shape[-3:])

            rgb_patch_features[i] = current_rgb_patches
            depth_patch_features[i] = current_depth_patches

        # Preprocess and aggregate patches
        rgb_patch_features = [x.reshape(-1, *x.shape[-3:]) for x in rgb_patch_features]
        rgb_patch_features = self.forward_modules["preprocessing"](rgb_patch_features)
        rgb_patch_features = self.forward_modules["preadapt_aggregator"](rgb_patch_features)

        depth_patch_features = [x.reshape(-1, *x.shape[-3:]) for x in depth_patch_features]
        depth_patch_features = self.forward_modules["preprocessing"](depth_patch_features)
        depth_patch_features = self.forward_modules["preadapt_aggregator"](depth_patch_features)

        return rgb_patch_features, patch_shapes, depth_patch_features

    def trainer(self, training_data, val_data, name):
        state_dict = {}
        ckpt_path = glob.glob(self.ckpt_dir + '/ckpt_best*')
        ckpt_path_save = os.path.join(self.ckpt_dir, "ckpt.pth")
        if len(ckpt_path) != 0:
            LOGGER.info("Start testing, ckpt file found!")
            return 0., 0., 0., 0., 0., -1.

        def update_state_dict():
            state_dict["discriminator"] = OrderedDict({
                k: v.detach().cpu()
                for k, v in self.discriminator.state_dict().items()})
            if self.pre_proj > 0:
                state_dict["pre_projection"] = OrderedDict({
                    k: v.detach().cpu()
                    for k, v in self.pre_projection.state_dict().items()})


        progress_bar = tqdm.tqdm(range(self.meta_epochs), unit='epoch')
        progress_bar_string = ""
        best_record = None
        for i_epoch in progress_bar:
            self.forward_modules.eval()

            progress_bar_str, pt, pf = self._train_discriminator(training_data, i_epoch, progress_bar, progress_bar_string)
            update_state_dict()

            if (i_epoch + 1) % self.eval_epochs == 0:
                images, scores, segmentations, labels_gt, masks_gt = self.predict(val_data)
                image_auroc, image_ap, pixel_auroc, pixel_ap, pixel_pro = self._evaluate(images, scores, segmentations,
                                                                                         labels_gt, masks_gt, name)

                self.logger.logger.add_scalar("i-auroc", image_auroc, i_epoch)
                self.logger.logger.add_scalar("i-ap", image_ap, i_epoch)
                self.logger.logger.add_scalar("p-auroc", pixel_auroc, i_epoch)
                self.logger.logger.add_scalar("p-ap", pixel_ap, i_epoch)
                self.logger.logger.add_scalar("p-pro", pixel_pro, i_epoch)

                eval_path = './results/eval/' + name + '/'
                train_path = './results/training/' + name + '/'
                if best_record is None:
                    best_record = [image_auroc, image_ap, pixel_auroc, pixel_ap, pixel_pro, i_epoch]
                    ckpt_path_best = os.path.join(self.ckpt_dir, str(image_auroc) + "_" + str(pixel_auroc)+"_ckpt_best_{}.pth".format(i_epoch))
                    torch.save(state_dict, ckpt_path_best)
                    shutil.rmtree(eval_path, ignore_errors=True)
                    shutil.copytree(train_path, eval_path)

                elif image_auroc + pixel_auroc > best_record[0] + best_record[2]:
                    best_record = [image_auroc, image_ap, pixel_auroc, pixel_ap, pixel_pro, i_epoch]
                    os.remove(ckpt_path_best)
                    ckpt_path_best = os.path.join(self.ckpt_dir, str(image_auroc) + "_" + str(pixel_auroc)+"_ckpt_best_{}.pth".format(i_epoch))
                    torch.save(state_dict, ckpt_path_best)
                    shutil.rmtree(eval_path, ignore_errors=True)
                    shutil.copytree(train_path, eval_path)

                progress_bar_string = (
                    f" IAUC:{round(image_auroc * 100, 2)}({round(best_record[0] * 100, 2)})"
                    f" IAP:{round(image_ap * 100, 2)}({round(best_record[1] * 100, 2)})"
                    f" PAUC:{round(pixel_auroc * 100, 2)}({round(best_record[2] * 100, 2)})"
                    f" PAP:{round(pixel_ap * 100, 2)}({round(best_record[3] * 100, 2)})"
                    f" PRO:{round(pixel_pro * 100, 2)}({round(best_record[4] * 100, 2)})"
                    f" E:{i_epoch}({best_record[-1]})"
                )
                progress_bar_str += progress_bar_string
                progress_bar.set_description_str(progress_bar_str)

            torch.save(state_dict, ckpt_path_save)
        return best_record

    def _train_discriminator(self, input_data, current_epoch, progress_bar, progress_bar_string):
        """Train the discriminator with multi-scale noise augmentation.

        Args:
            input_data: Training data loader
            current_epoch: Current epoch number
            progress_bar: Progress bar object
            progress_bar_string: string for progress bar

        Returns:
            Tuple of (progress_string, mean_true_prob, mean_fake_prob)
        """
        self.forward_modules.eval()
        if self.pre_proj > 0:
            self.pre_projection.train()
        self.discriminator.train()

        total_losses = []
        sample_count = 0

        for iteration, data_item in enumerate(input_data):
            self.dsc_opt.zero_grad()
            if self.pre_proj > 0:
                self.proj_opt.zero_grad()

            # Load data
            augmented_image = data_item["augmented_image"].to(torch.float).to(self.device)
            clean_image = data_item["image"].to(torch.float).to(self.device)
            augmented_depth = data_item["augmented_depth"].to(torch.float).to(self.device)
            clean_depth = data_item["depth"].to(torch.float).to(self.device)

            if self.pre_proj > 0:
                # High-scale noise (scale=6) on input images
                noise_rgb_high, noise_depth_high = self.get_rgb_depth_noise(6, clean_image, clean_depth)
                noisy_image_high = clean_image + noise_rgb_high
                noisy_depth_high = clean_depth + noise_depth_high
                noisy_rgb_features_high, _, noisy_depth_features_high = self._embed(
                    noisy_image_high, depths=noisy_depth_high, evaluation=False
                )
                noisy_features_high = torch.cat((noisy_rgb_features_high, noisy_depth_features_high), dim=1)
                noisy_features_high = self.pre_projection(noisy_features_high)

                # Extract features from augmented images
                augmented_rgb_features, _, augmented_depth_features = self._embed(
                    augmented_image, depths=augmented_depth, evaluation=False
                )
                augmented_features = torch.cat((augmented_rgb_features, augmented_depth_features), dim=1)
                augmented_features = self.pre_projection(augmented_features)

                # Extract clean features
                clean_rgb_features, _, clean_depth_features = self._embed(
                    clean_image, depths=clean_depth, evaluation=False
                )

                # Medium-scale noise (scale=2) on features
                noise_rgb_medium, noise_depth_medium = self.get_rgb_depth_noise(
                    2, clean_rgb_features, clean_depth_features
                )
                noisy_rgb_features_medium = clean_rgb_features + noise_rgb_medium
                noisy_depth_features_medium = clean_depth_features + noise_depth_medium
                noisy_features_medium = torch.cat((noisy_rgb_features_medium, noisy_depth_features_medium), dim=1)
                noisy_features_medium = self.pre_projection(noisy_features_medium)

                # Combine clean features
                clean_features = torch.cat((clean_rgb_features, clean_depth_features), dim=1)
                clean_features = self.pre_projection(clean_features)

                # Low-scale noise (scale=1) on projected features
                noise_low = torch.normal(0, self.noise, clean_features.shape).to(self.device)
                noisy_features_low = clean_features + noise_low
            else:
                print("****************")

            mask_ground_truth = data_item["small_mask"].reshape(-1, 1).to(self.device)


            combined_scores_high = self.discriminator(torch.cat([clean_features, noisy_features_high]))
            clean_scores_high = combined_scores_high[:len(clean_features)]
            noisy_scores_high = combined_scores_high[len(clean_features):]
            clean_loss_high = torch.nn.BCELoss()(clean_scores_high, torch.zeros_like(clean_scores_high))
            noisy_loss_high = torch.nn.BCELoss()(noisy_scores_high, torch.ones_like(noisy_scores_high))
            bce_loss_high = clean_loss_high + noisy_loss_high

            combined_scores_medium = self.discriminator(torch.cat([clean_features, noisy_features_medium]))
            clean_scores_medium = combined_scores_medium[:len(clean_features)]
            noisy_scores_medium = combined_scores_medium[len(clean_features):]
            clean_loss_medium = torch.nn.BCELoss()(clean_scores_medium, torch.zeros_like(clean_scores_medium))
            noisy_loss_medium = torch.nn.BCELoss()(noisy_scores_medium, torch.ones_like(noisy_scores_medium))
            bce_loss_medium = clean_loss_medium + noisy_loss_medium

            combined_scores_low = self.discriminator(torch.cat([clean_features, noisy_features_low]))
            clean_scores_low = combined_scores_low[:len(clean_features)]
            noisy_scores_low = combined_scores_low[len(clean_features):]
            clean_loss_low = torch.nn.BCELoss()(clean_scores_low, torch.zeros_like(clean_scores_low))
            noisy_loss_low = torch.nn.BCELoss()(noisy_scores_low, torch.ones_like(noisy_scores_low))
            bce_loss_low = clean_loss_low + noisy_loss_low

            total_bce_loss = bce_loss_high + bce_loss_medium + bce_loss_low





            augmented_scores = self.discriminator(augmented_features)
            focal_output = torch.cat([1 - augmented_scores, augmented_scores], dim=1)
            focal_loss = self.focal_loss(focal_output, mask_ground_truth)

            total_loss = total_bce_loss + 1.0 * focal_loss
            total_loss.backward()
            if self.pre_proj > 0:
                self.proj_opt.step()
            if self.train_backbone:
                self.backbone_opt.step()
            self.dsc_opt.step()

            self.logger.logger.add_scalar("loss", total_loss, self.logger.g_iter)
            self.logger.step()

            total_losses.append(total_loss.detach().cpu().item())

            total_losses_ = np.mean(total_losses)
            sample_count = sample_count + clean_image.shape[0]

            # Update progress bar
            progress_bar_description = f"epoch:{current_epoch} loss:{total_losses_:.2e}"
            progress_bar_description += f" sample:{sample_count}"
            final_progress_bar_string = progress_bar_description
            final_progress_bar_string += progress_bar_string
            progress_bar.set_description_str(final_progress_bar_string)

            if sample_count > self.limit:
                break

        return final_progress_bar_string, None, None

    def tester(self, test_data, name):
        ckpt_path = glob.glob(self.ckpt_dir + '/ckpt_best*')
        if len(ckpt_path) != 0:
            state_dict = torch.load(ckpt_path[0], map_location=self.device)
            if 'discriminator' in state_dict:
                self.discriminator.load_state_dict(state_dict['discriminator'])
                if "pre_projection" in state_dict:
                    self.pre_projection.load_state_dict(state_dict["pre_projection"])
            else:
                self.load_state_dict(state_dict, strict=False)

            images, scores, segmentations, labels_gt, masks_gt = self.predict(test_data)
            image_auroc, image_ap, pixel_auroc, pixel_ap, pixel_pro = self._evaluate(images, scores, segmentations,
                                                                                     labels_gt, masks_gt, name, path='eval')
            epoch = int(ckpt_path[0].split('_')[-1].split('.')[0])
        else:
            image_auroc, image_ap, pixel_auroc, pixel_ap, pixel_pro, epoch = 0., 0., 0., 0., 0., -1.
            LOGGER.info("No ckpt file found!")

        return image_auroc, image_ap, pixel_auroc, pixel_ap, pixel_pro, epoch

    def _evaluate(self, images, scores, segmentations, labels_gt, masks_gt, name, path='training'):
        scores = np.squeeze(np.array(scores))
        image_min_scores = min(scores)
        image_max_scores = max(scores)
        norm_scores = (scores - image_min_scores) / (image_max_scores - image_min_scores + 1e-10)

        image_scores = metrics.compute_imagewise_retrieval_metrics(norm_scores, labels_gt, path)
        image_auroc = image_scores["auroc"]
        image_ap = image_scores["ap"]

        if len(masks_gt) > 0:
            segmentations = np.array(segmentations)
            min_scores = np.min(segmentations)
            max_scores = np.max(segmentations)
            norm_segmentations = (segmentations - min_scores) / (max_scores - min_scores + 1e-10)

            pixel_scores = metrics.compute_pixelwise_retrieval_metrics(norm_segmentations, masks_gt, path)
            pixel_auroc = pixel_scores["auroc"]
            pixel_ap = pixel_scores["ap"]
            if path == 'eval':
                try:
                    pixel_pro = metrics.compute_pro(np.squeeze(np.array(masks_gt)), norm_segmentations)

                except:
                    pixel_pro = 0.
            else:
                pixel_pro = 0.

        else:
            pixel_auroc = -1.
            pixel_ap = -1.
            pixel_pro = -1.
            return image_auroc, image_ap, pixel_auroc, pixel_ap, pixel_pro

        defects = np.array(images)
        targets = np.array(masks_gt)
        for i in range(len(defects)):
            defect = utils.torch_format_2_numpy_img(defects[i])
            target = utils.torch_format_2_numpy_img(targets[i])

            mask = cv2.cvtColor(cv2.resize(norm_segmentations[i], (defect.shape[1], defect.shape[0])),
                                cv2.COLOR_GRAY2BGR)
            mask = (mask * 255).astype('uint8')
            mask = cv2.applyColorMap(mask, cv2.COLORMAP_JET)

            image_up = np.hstack([defect, target, mask])
            image_up = cv2.resize(image_up, (256 * 3, 256))
            full_path = './results/' + path + '/' + name + '/'
            utils.del_remake_dir(full_path, del_flag=False)
            cv2.imwrite(full_path + str(i + 1).zfill(3) + '.png', image_up)

        return image_auroc, image_ap, pixel_auroc, pixel_ap, pixel_pro

    def predict(self, test_dataloader):
        """This function provides anomaly scores/maps for full dataloaders."""
        self.forward_modules.eval()

        image_paths = []
        images = []
        depths = []
        scores = []
        masks = []
        labels_gt = []
        masks_gt = []

        with tqdm.tqdm(test_dataloader, desc="Inferring...", leave=False, unit='batch') as data_iterator:
            for data in data_iterator:
                if isinstance(data, dict):
                    labels_gt.extend(data["is_anomaly"].numpy().tolist())
                    if data.get("ground_truth_mask", None) is not None:
                        masks_gt.extend(data["ground_truth_mask"].numpy().tolist())
                    image = data["image"]
                    depth = data["depth"]
                    images.extend(image.numpy().tolist())
                    depths.extend(image.numpy().tolist())
                    image_paths.extend(data["image_path"])
                _scores, _masks = self._predict(image, depth)
                for score, mask in zip(_scores, _masks):
                    scores.append(score)
                    masks.append(mask)

        return images, scores, masks, labels_gt, masks_gt

    def _predict(self, image, depth):
        """Infer score and mask for a batch of images."""
        image = image.to(torch.float).to(self.device)
        depth = depth.to(torch.float).to(self.device)
        self.forward_modules.eval()

        if self.pre_proj > 0:
            self.pre_projection.eval()
        self.discriminator.eval()

        with torch.no_grad():

            patch_features, patch_shapes, depth_features = self._embed(image,depths=depth, provide_patch_shapes=True, evaluation=True)
            patch_features = torch.cat((patch_features,depth_features),dim=1)
            if self.pre_proj > 0:
                patch_features = self.pre_projection(patch_features)
                patch_features = patch_features[0] if len(patch_features) == 2 else patch_features



            patch_scores = image_scores = self.discriminator(patch_features)
            patch_scores = self.patch_maker.unpatch_scores(patch_scores, batchsize=image.shape[0])
            scales = patch_shapes[0]
            patch_scores = patch_scores.reshape(image.shape[0], scales[0], scales[1])
            masks = self.anomaly_segmentor.convert_to_segmentation(patch_scores)

            image_scores = self.patch_maker.unpatch_scores(image_scores, batchsize=image.shape[0])
            image_scores = self.patch_maker.score(image_scores)
            if isinstance(image_scores, torch.Tensor):
                image_scores = image_scores.cpu().numpy()

        return list(image_scores), list(masks)
