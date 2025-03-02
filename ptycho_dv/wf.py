import torch
from ptycho_dv.dataset import img2patch, fwd_patches_complex


def wf_iteration(cur_est, probe, y_meas, patch_bounds, discretized_sys_mat, prm=1):
    """
    Revises the estimate of a complex object using the WF (Wavefront) method.

    Args:
        cur_est (Tensor): The current estimate (initialization).
        patch_bounds (Tensor): Array containing the coordinates of the top-left and bottom-right corners of the patches.
        probe (Tensor): The probe array.
        y_meas (Tensor): The raw measurements.
        discretized_sys_mat (Tensor): The eigenvalue used to determine the step size.
        prm (float, optional): A parameter that is set to 1 when the Fourier transform (FT) is orthonormal. Default is 1.

    Returns:
        Tensor: The revised estimate of the complex object.
    """
    
    patch = img2patch(cur_est, patch_bounds)

    f = fwd_patches_complex(patch, probe)

    inv_f = f - y_meas * torch.exp(1j * torch.angle(f))

    inv_f = torch.fft.fftshift(inv_f, dim=(-2, -1))
    inv_f = torch.fft.ifft2(inv_f, norm='ortho')
    inv_f = torch.fft.ifftshift(inv_f, dim=(-2, -1))

    output = cur_est - patch2img(inv_f * torch.conj(probe), patch_bounds, cur_est.shape, norm=None) / torch.max(prm * discretized_sys_mat)

    return output


def patch2img(img_patch, coords, img_sz, norm=None):
    """
    Converts patches of an image into a larger image based on the coordinates of the patches. 
    For overlapping areas, the sum of the values is used.

    Args:
        img_patch (Tensor): The input image patches.
        coords (Tensor): The coordinates of the patches in the target large image.
        img_sz (tuple): The size of the underlying ground truth image.
        norm (optional): A normalization parameter (not used in the current function).

    Returns:
        Tensor: The reconstructed image.
    """

    reference = torch.zeros(img_sz, dtype=img_patch.dtype, device=img_patch.device)

    for i in range(coords.shape[0]):
        reference[coords[i][0]:coords[i][1], coords[i][2]:coords[i][3]] += img_patch[i]

    return reference
