import torch
import torchvision.transforms.functional as TF
import numpy as np
import torch.nn.functional as F


class RGBDAugmentor:
    """ perform augmentation on RGB-D video """

    def __init__(self, crop_size):
        self.crop_size = crop_size
        self.max_scale = 0.5

        # ColorJitter parameters (same as original)
        self.brightness = 0.4
        self.contrast = 0.4
        self.saturation = 0.4
        self.hue = 0.2 / 3.14

    def spatial_transform(self, images, depths, poses, intrinsics):
        """ cropping and resizing with random crop augmentation """
        ht, wd = images.shape[2:]

        # Calculate minimum scale to ensure image is larger than crop_size
        min_scale = max(self.crop_size[0] / ht, self.crop_size[1] / wd)
        min_scale = max(min_scale, 1.0)  # At least scale 1.0

        # Random scale (80% of the time scale up, always ensure >= min_scale)
        scale = min_scale
        if np.random.rand() < 0.8:
            scale = 2 ** np.random.uniform(np.log2(min_scale), np.log2(min_scale) + self.max_scale)

        intrinsics = scale * intrinsics

        ht1 = int(scale * ht)
        wd1 = int(scale * wd)

        depths = depths.unsqueeze(dim=1)

        images = F.interpolate(images, (ht1, wd1), mode='bicubic', align_corners=False)
        depths = F.interpolate(depths, (ht1, wd1), recompute_scale_factor=False)

        # Random crop (instead of center crop)
        max_y0 = max(0, images.shape[2] - self.crop_size[0])
        max_x0 = max(0, images.shape[3] - self.crop_size[1])

        y0 = np.random.randint(0, max_y0 + 1) if max_y0 > 0 else 0
        x0 = np.random.randint(0, max_x0 + 1) if max_x0 > 0 else 0

        # Adjust intrinsics: cx, cy need to be shifted by crop offset
        intrinsics = intrinsics - torch.tensor([0.0, 0.0, x0, y0])
        images = images[:, :, y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        depths = depths[:, :, y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]

        depths = depths.squeeze(dim=1)
        return images, poses, depths, intrinsics

    def color_transform(self, images):
        """ color jittering using native torch operations (no PIL conversion) """
        num, ch, ht, wd = images.shape

        # Reshape: [N, C, H, W] -> [C, H, W*N] for batch processing
        images = images.permute(1, 2, 3, 0).reshape(ch, ht, wd * num)

        # Convert BGR to RGB and normalize to [0, 1]
        images = images[[2, 1, 0]] / 255.0

        # Apply ColorJitter in random order (same as torchvision.transforms.ColorJitter)
        fn_idx = torch.randperm(4)
        for fn_id in fn_idx:
            if fn_id == 0:
                # Brightness
                brightness_factor = 1.0 + np.random.uniform(-self.brightness, self.brightness)
                images = TF.adjust_brightness(images, brightness_factor)
            elif fn_id == 1:
                # Contrast
                contrast_factor = 1.0 + np.random.uniform(-self.contrast, self.contrast)
                images = TF.adjust_contrast(images, contrast_factor)
            elif fn_id == 2:
                # Saturation
                saturation_factor = 1.0 + np.random.uniform(-self.saturation, self.saturation)
                images = TF.adjust_saturation(images, saturation_factor)
            elif fn_id == 3:
                # Hue
                hue_factor = np.random.uniform(-self.hue, self.hue)
                images = TF.adjust_hue(images, hue_factor)

        # RandomGrayscale (p=0.1)
        if np.random.rand() < 0.1:
            images = TF.rgb_to_grayscale(images, num_output_channels=3)

        # RandomInvert (p=0.1)
        if np.random.rand() < 0.1:
            images = TF.invert(images)

        # Convert RGB back to BGR and scale to [0, 255]
        images = images[[2, 1, 0]] * 255.0

        # Reshape back: [C, H, W*N] -> [N, C, H, W]
        return images.reshape(ch, ht, wd, num).permute(3, 0, 1, 2).contiguous()

    def __call__(self, images, poses, depths, intrinsics):
        if np.random.rand() < 0.5:
            images = self.color_transform(images)

        return self.spatial_transform(images, depths, poses, intrinsics)
