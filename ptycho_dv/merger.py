from torch import nn
import torch
from einops import rearrange
from ptycho_dv.net_layers import UNet
from ptycho_dv.wf import wf_iteration, patch2img


def padding_patches_average(reference_size, scanLoc, image_patches):
    """
    Converts image patches into a larger image based on their coordinates in the target image. 
    For overlapping regions, the pixel values are averaged.

    Args:
        reference_size (tuple): Size of the target large image (height, width).
        scanLoc (torch.Tensor): Coordinates of the patches in the target image.
        image_patches (torch.Tensor): Input patches of the image.

    Returns:
        torch.Tensor: The reconstructed image.
        torch.Tensor: A mask indicating non-zero regions in the reconstructed image.
    """

    assert image_patches.dim() == 3
    num_patches, width_patch, height_patch = image_patches.shape

    reference = torch.stack([
        nn.functional.pad(image_patches[i], pad=(
            scanLoc[i][2], reference_size[1] - height_patch - scanLoc[i][2],
            scanLoc[i][0], reference_size[0] - width_patch - scanLoc[i][0]
        )) for i in range(num_patches)
    ])

    count_nonzero = torch.count_nonzero(reference, 0)
    reference = torch.sum(reference, 0)
    reference[reference != 0] /= count_nonzero[reference != 0]

    updated_mask = torch.zeros(reference_size, dtype=torch.float32, device=image_patches.device)
    updated_mask[reference != 0] = 1

    reference = torch.unsqueeze(reference, 0)

    return reference, updated_mask


class PtychoDUMerger(nn.Module):
    def __init__(
            self,
            is_use_cnn,
            iteration,
    ):
        """
        Initializes the PtychoDUMerger module for deep unfolding.

        Args:
            is_use_cnn (bool): Flag indicating whether to use a CNN for refinement during unfolding.
            iteration (int): Number of iterations in the deep unfolding process.
        """
    
        super().__init__()

        self.is_use_cnn = is_use_cnn
        self.iteration = iteration

        if self.is_use_cnn:
            self.cnn = UNet(
                dimension=2,
                i_nc=2,
                o_nc=2,
                f_root=64,
                conv_times=3,
                is_bn=False,
                activation='relu',
                is_residual=False,
                up_down_times=3,
                is_spe_norm=False,
                padding=(0, 0)
            )

    def forward(self, patches, scanLoc, reference_size, probe, noisy_data):
        """
        Executes the forward process of the deep unfolding module.

        Args:
            patches (torch.Tensor): Input patches of the reconstructed image.
            scanLoc (torch.Tensor): Scan locations representing the coordinates of patches in the large image.
            reference_size (tuple): Size of the target ground-truth image (height, width).
            probe (torch.Tensor): Probe array used in the forward model.
            noisy_data (torch.Tensor): Raw measurements from the imaging process.

        Returns:
            torch.Tensor: The fused image after deep unfolding.
            torch.Tensor: An updated mask indicating the valid regions of the fused image.
        """
    
        obj_wgt_mat = []
        for i in range(patches.shape[0]):
            obj_wgt_mat.append(
                patch2img(torch.abs(torch.stack([torch.view_as_complex(probe[i])] * patches[i].shape[0], 0)) ** 2, scanLoc[i], reference_size))
        obj_wgt_mat = torch.stack(obj_wgt_mat, 0)

        patches = patches[..., 0] + patches[..., 1] * 1j

        images, updated_masks = [], []
        for i in range(patches.shape[0]):
            tmp_image, tmp_updated_mask = padding_patches_average(reference_size, scanLoc[i], patches[i])

            images.append(tmp_image)
            updated_masks.append(tmp_updated_mask)

        images, updated_masks = [torch.stack(i, dim=0) for i in [images, updated_masks]]

        for _ in range(self.iteration):

            images_updated = []
            for i in range(patches.shape[0]):
                images_updated.append(
                    wf_iteration(images[i].squeeze(0), torch.view_as_complex(probe[i]), noisy_data[i].squeeze(-1), scanLoc[i], obj_wgt_mat[i]).unsqueeze(0)
                )

            images = torch.stack(images_updated, 0)

            if self.is_use_cnn:
                images = torch.view_as_real(images)
                images = rearrange(images, 'b n w h c -> b (n c) w h')

                images = self.cnn(images)

                images = rearrange(images, 'b c w h -> b w h c')
                images = images[..., 0] + images[..., 1] * 1j

            else:
                images = images.squeeze(1)

        fused_images = torch.view_as_real(images)

        updated_masks_new = torch.zeros_like(updated_masks)
        updated_masks_new[:, 128: 256, 128: 256] = 1

        return fused_images, updated_masks_new
