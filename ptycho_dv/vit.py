import torch
from torch import nn
import math
import pytorch_lightning as pl
from ptycho_dv.dataset import fwd_patches_complex, fwd_complex
from ptycho_dv.net_layers import pair, Transformer
from ptycho_dv.merger import PtychoDUMerger


class PtyChoViT(nn.Module):
    def __init__(self, image_size, patch_size, dim, depth, heads, mlp_dim, dim_head, channels=1,
                 dropout=0., L=10, is_pos_embedding=True, is_seperated=False, is_blind=False):
        """
        Initializes the Vision Transformer (ViT) module for the PtychoDV model.

        Args:
            image_size (int): Size of the underlying ground truth image.
            patch_size (int): Size of the measurement patches.
            dim (int): The dimension of the embedding layer.
            depth (int): The number of repetitions of the attention module in the transformer.
            heads (int): The number of attention heads in the transformer.
            mlp_dim (int): The dimension of the hidden layer in the linear layer of the transformer.
            dim_head (int): The dimension of the data in each attention head of the transformer.
            channels (int, optional): The number of input channels. Default is 1.
            dropout (float, optional): Dropout rate. Default is 0.
            L (int, optional): The number of positional encoding layers. Default is 10.
            is_pos_embedding (bool, optional): Flag to control whether positional embedding is applied. Default is True.
            is_seperated (bool, optional): Flag to control whether to use separate decoders for the real and imaginary parts. Default is False.
            is_blind (bool, optional): Flag to control whether the probe is known (is_blind=False) or not (is_blind=True). Default is False.
        """

        super().__init__()

        self.L = L
        self.patch_size = patch_size
        self.image_size = float(image_size)
        self.is_pos_embedding = is_pos_embedding
        self.is_seperated = is_seperated
        self.is_blind = is_blind

        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        input_dim = channels * patch_height * patch_width
        patch_dim = patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            nn.Linear(input_dim, dim * 2),
        )

        self.embedding_to_patch = nn.Linear(dim * 2, patch_dim * 2)

        self.to_pos_embedding = nn.Sequential(
            nn.Linear(self.L * 4, dim * 2)
        )

        self.transformer = Transformer(
            dim=dim * 2,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            mlp_dim=mlp_dim * 2,
            dropout=dropout
        )

        if not self.is_blind:
            self.to_probe_embedding = nn.Linear(2 * patch_height * patch_width, dim * 2)

    def forward(self, x, scanLoc, probe=None):
        """
        Forward pass of the Vision Transformer module.

        Args:
            x (Tensor): Input tensor.
            scanLoc (Tensor): Coordinates of all probe locations.
            probe (Tensor, optional): Probe array. Only required if 'is_blind' is False.

        Returns:
            Tensor: The output after processing through the transformer.
        """
                
        scanLoc = scanLoc[..., [0, 2]].to(torch.float32)
        scanLoc = scanLoc / self.image_size

        # positional encoding
        for l in range(self.L):

            cur_freq = torch.cat(
                [torch.sin(2 ** l * math.pi * scanLoc), torch.cos(2 ** l * math.pi * scanLoc)], dim=-1)

            if l == 0:
                tot_freq = cur_freq
            else:
                tot_freq = torch.cat([tot_freq, cur_freq], dim=-1)

        scanLoc = tot_freq

        b, num_patch = x.shape[0], x.shape[1]
        x = x.reshape([b, num_patch, -1])

        x = self.to_patch_embedding(x)

        if self.is_pos_embedding:
            x = x + self.to_pos_embedding(scanLoc)

        if not self.is_blind:
            assert probe is not None

            probe = torch.unsqueeze(probe, 1)
            probe = probe.reshape([b, 1, -1])

            probe = self.to_probe_embedding(probe)

            x = torch.concat([probe, x], 1)

        if not self.is_seperated:
            x = self.transformer(x)
            x = self.embedding_to_patch(x)

            if not self.is_blind:
                x = x[:, 1:]

            x = x.reshape([b, num_patch, self.patch_size, self.patch_size, 2])

        else:
            x_real = self.embedding_to_patch_real(self.transformer_real(x))
            x_real = x_real.reshape([b, num_patch, self.patch_size, self.patch_size])

            x_imag = self.embedding_to_patch_imag(self.transformer_imag(x))
            x_imag = x_imag.reshape([b, num_patch, self.patch_size, self.patch_size])

            x = torch.stack([x_real, x_imag], -1)

        return x


class PtyChoModule(pl.LightningModule):
    def __init__(self, config):
        """
        Initializes the PtychoDV module.

        Args:
            config (dict): A dictionary containing all pre-defined configurations.
        """

        super().__init__()

        self.config = config

        head_dict = {
            'vit': lambda: PtyChoViT(
                        image_size=config['method']['ViT']['image_size'],
                        patch_size=config['method']['ViT']['patch_size'],
                        dim=config['method']['ViT']['dim'],
                        depth=config['method']['ViT']['depth'],
                        heads=config['method']['ViT']['heads'],
                        mlp_dim=config['method']['ViT']['dim'],
                        dim_head=config['method']['ViT']['dim_head'],
                        channels=1,
                        dropout=0.,
                        L=10,
                        is_pos_embedding=True,
                        is_seperated=False,
                        is_blind=True
                    ),
        }

        merger_dict = {
            'du': lambda: PtychoDUMerger(
                is_use_cnn=config['method']['merger_du']['is_use_cnn'],
                iteration=config['method']['merger_du']['iteration'],
            ),
        }

        self.head = head_dict['vit']()
        self.merger = merger_dict['du']()

        loss_dict = {
            'l2': nn.MSELoss,
            'l1': nn.L1Loss
        }

        self.loss_fn = loss_dict[config['method']['loss']['fn']]()

        self.log_init_val_images = False

        self.reference_size = (config['method']['ViT']['image_size'], config['method']['ViT']['image_size'])

    def forward(self, x, scanLoc, probe=None, y_meas=None):
        """
        Forward pass of the PtychoDV module.

        Args:
            x (Tensor): Input tensor.
            scanLoc (Tensor): Coordinates of all probe locations.
            probe (Tensor, optional): Probe array.
            y_meas (Tensor, optional): Raw measurement data.

        Returns:
            Tuple[Tensor, Tensor, Tensor]: The reconstructed patches from ViT, final reconstructed images, and updated masks indicating the non-zero regions of the reconstructed image.
        """

        patches_hat = self.head(x, scanLoc, probe)

        images_hat, updated_masks = self.merger(patches_hat, scanLoc, self.reference_size, probe, y_meas)

        return patches_hat, images_hat, updated_masks

    def training_step(self, batch, batch_idx):
        """
        Built-in function of pytorch-lightening.
        """

        # config
        coff_loss_gt_patch = self.config['method']['loss']['coff_loss_gt_patch']
        coff_loss_fwd_patch = self.config['method']['loss']['coff_loss_fwd_patch']
        coff_loss_fwd = self.config['method']['loss']['coff_loss_fwd']
        coff_loss_gt = self.config['method']['loss']['coff_loss_gt']

        mode = self.config['method']['loss']['mode']

        noisy_data, scan_coords, sample, probe, sample_patches = batch

        sample_patches_hat, sample_hat, updated_masks = self(noisy_data, scan_coords, probe, noisy_data)

        loss_gt_patch = self.loss_helper(sample_patches_hat, sample_patches, mode=mode)

        if coff_loss_fwd_patch > 0:
            # patch-wise loss function
            noisy_data_patches_hat = torch.stack([fwd_patches_complex(
                obj_patches=sample_patches_hat[i, ..., 0] + sample_patches_hat[i, ..., 1] * 1j,
                probe=torch.view_as_complex(probe)[i],
            ) for i in range(sample_patches_hat.shape[0])], dim=0).abs().unsqueeze(-1)

            if self.config['dataset']['is_problem0']:
                noisy_data = torch.view_as_complex(noisy_data)
                noisy_data = abs(noisy_data)

            loss_fwd_patch = self.loss_helper(noisy_data_patches_hat, noisy_data)

        else:
            loss_fwd_patch = loss_gt_patch

        loss_gt = self.loss_helper(sample_hat * updated_masks.unsqueeze(-1), sample * updated_masks.unsqueeze(-1), mode=mode)

        if coff_loss_fwd > 0:
            # image-wise loss function
            sample_hat_to_meas = torch.stack([fwd_complex(
                obj=sample_hat[i, ..., 0] + sample_hat[i, ..., 1] * 1j,
                probe=torch.view_as_complex(probe)[i],
                coords=scan_coords[i]
            ) for i in range(sample_patches_hat.shape[0])], dim=0)

            sample_to_meas = torch.stack([fwd_complex(
                obj=torch.view_as_complex(sample)[i],
                probe=torch.view_as_complex(probe)[i],
                coords=scan_coords[i]
            ) for i in range(sample_patches_hat.shape[0])], dim=0)

            loss_fwd = self.loss_helper(abs(sample_to_meas), abs(sample_hat_to_meas))

        else:
            loss_fwd = loss_gt

        # logging
        loss = loss_gt_patch * coff_loss_gt_patch + \
            loss_fwd_patch * coff_loss_fwd_patch + \
            loss_fwd * coff_loss_fwd + \
            loss_gt * coff_loss_gt

        cut_off = probe.shape[-2] // 2
        self.log(name='tra_mse', value=torch.nn.functional.mse_loss(
            sample[:, cut_off:-cut_off, cut_off:-cut_off], sample_hat[:, cut_off:-cut_off, cut_off:-cut_off]
        ), on_epoch=True, on_step=False)

        self.log(name='loss_gt_patch', value=loss_gt_patch)
        self.log(name='loss_fwd_patch', value=loss_fwd_patch)
        self.log(name='loss_fwd', value=loss_fwd)
        self.log(name='loss_gt', value=loss_gt)
        self.log(name='loss', value=loss)

        return loss

    def validation_step(self, batch, batch_idx):
        """
        Built-in function of pytorch-lightening.
        """

        noisy_data, scan_coords, sample, probe, sample_patches = batch

        _, sample_hat, _ = self(noisy_data, scan_coords, probe, noisy_data)

        cut_off = probe.shape[-2] // 2
        self.log(name='val_mse', value=torch.nn.functional.mse_loss(
            sample[:, cut_off:-cut_off, cut_off:-cut_off], sample_hat[:, cut_off:-cut_off, cut_off:-cut_off]
        ))

    def configure_optimizers(self):
        """
        Built-in function of pytorch-lightening.
        """
                
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config['train']['lr'])
        return optimizer

    def loss_helper(self, pre, tar, mode='real_imag'):
        """
        A helper function to compute the loss between predicted and target tensors.

        Args:
            pre (Tensor): The predicted tensor.
            tar (Tensor): The target tensor.
            mode (str, optional): The mode used for loss computation. Default is 'real_imag'.

        Returns:
            Tensor: The computed loss.
        """

        return self.loss_fn(pre, tar)
