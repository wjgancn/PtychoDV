import h5py
import tqdm
from torch.utils import data
import os
import numpy as np
from ptycho_pmace.utils.utils import gen_scan_loc, get_proj_coords_from_data
import torch
import io


def read_dat(path, dtype):
    """
    Reads a numpy array from a .dat file.

    Args:
        path (str): The path to the .dat file.
        dtype (numpy.dtype): The data type of the array elements.

    Returns:
        numpy.ndarray: The loaded array data.
    """

    f = io.open(path, 'rb')
    rev = f.read()

    rev = np.frombuffer(rev, dtype=dtype)
    rev_size = rev.shape[0]

    rev = np.reshape(rev, (int(rev_size ** 0.5), int(rev_size ** 0.5)))
    rev = np.ascontiguousarray(rev)

    if dtype == complex:
        rev = rev.astype(np.complex64)
    elif dtype == float:
        rev = rev.astype(np.float32)

    return rev


class WaveProDataset(data.Dataset):
    @staticmethod
    def complex_average_rescale(x, scale):
        """
        Downscale or resize a complex image using average operator.

        Args:
            x (numpy.ndarray): A complex-valued image.
            scale (int): The downscale factor (should be <1).

        Returns:
            numpy.ndarray: The downscaled complex image.
        """

        if scale == 1:
            return x

        new = np.zeros(shape=[x.shape[0] // scale, x.shape[1] // scale], dtype=x.dtype)

        for i in range(x.shape[0] // scale):
            for j in range(x.shape[1] // scale):
                new[i, j] = np.sum(
                    x[scale * i: scale * (i + 1), scale * j: scale * (j + 1)]
                ) / (scale ** 2)

        return new

    @staticmethod
    def complex_nearest_rescale(x, scale):
        """
        Downscale or resize a complex image using nearest-neighbor interpolation.

        Args:
            x (numpy.ndarray): A complex-valued image.
            scale (int): The downscale factor (should be <1).

        Returns:
            numpy.ndarray: The downscaled complex image.
        """

        if scale == 1:
            return x

        new = np.zeros(shape=[x.shape[0] // scale, x.shape[1] // scale], dtype=x.dtype)

        for i in range(x.shape[0] // scale):
            for j in range(x.shape[1] // scale):
                new[i, j] = x[scale * i, scale * j]

        return new

    def __init__(self, mode, config):
        """
        Initializes the WaveProDataset class.

        Args:
            mode (str): The dataset mode, which must be one of 'tra' (training), 'val' (validation), or 'tst' (testing).
            config (dict): A dictionary containing predefined configuration settings.

        Returns:
            None
        """

        self.mode = mode
        self.config = config

        assert self.mode in ['tra', 'val', 'tst']

        self.start_index, self.end_index = self.config['dataset']['WavePro'][self.mode + '_indexes']

    def __len__(self):
        """
        Returns the number of samples in the dataset.

        Returns:
            int: The number of samples.
        """
         
        return self.end_index - self.start_index

    def __getitem__(self, item):
        """
        Retrieves a sample from the dataset.

        Args:
            item (int): The index of the sample.

        Returns:
            tuple: A tuple containing the noisy data, scan coordinates, sample, probe, and sample patches, denoted as {y_i}, {D_i}, x, P, {D_i x} 
        """

        item += self.start_index

        sample_path = os.path.join(self.config['dataset']['WavePro']['path'], '%.6d.h5' % item)
        with h5py.File(sample_path, 'r') as f:
            sample = f['sample'][:]
            sample = self.complex_nearest_rescale(sample, 2)

        probe_idx = self.config['dataset']['WavePro']['probe_idx_range']

        if probe_idx > 0:
            probe_path = os.path.join(self.config['dataset']['WavePro']['path'], '%.6d.h5' % probe_idx)
            with h5py.File(probe_path, 'r') as f:
                probe = f['probe'][:]

        probe = self.complex_average_rescale(probe, 2)

        num_meas = self.config['dataset']['WavePro']['num_meas_range']
        spa_meas = self.config['dataset']['WavePro']['spa_meas_range']

        y_meas = np.zeros((num_meas, probe.shape[0], probe.shape[1]), dtype=np.float64)

        randomization = True if self.mode == 'tra' else False
        scan_loc = gen_scan_loc(
            sample, probe, num_meas, spa_meas, randomization=randomization)

        scan_coords = get_proj_coords_from_data(scan_loc, y_meas)

        sample = torch.from_numpy(sample)
        probe = torch.from_numpy(probe)

        noisy_data = fwd_complex(
            sample, probe, scan_coords
        )

        noisy_data = torch.abs(noisy_data)

        if self.config['dataset']['WavePro']['is_add_noise']:
            num_pts, m, n = noisy_data.shape
            photon_rate = self.config['dataset']['WavePro']['photon_rate']
            shot_noise_pm = self.config['dataset']['WavePro']['shot_noise_pm']

            noiseless_data = noisy_data ** 2
            noiseless_data = noiseless_data.numpy()

            # get peak signal value
            peak_signal_val = np.amax(noiseless_data)
            # calculate expected photon rate given peak signal value and peak photon rate
            expected_photon_rate = noiseless_data * photon_rate / peak_signal_val
            # poisson random values realization
            meas_in_photon_ct = np.random.poisson(expected_photon_rate, (num_pts, m, n))
            # add dark current noise
            noisy_data = meas_in_photon_ct + np.random.poisson(lam=shot_noise_pm, size=(num_pts, m, n))
            noisy_data = np.sqrt(noisy_data)
            noisy_data = torch.from_numpy(noisy_data).to(torch.float32)

        noisy_data = noisy_data.unsqueeze(-1)

        sample_patches = img2patch(sample, scan_coords)

        scan_coords = torch.from_numpy(scan_coords).to(torch.int32)
        sample = torch.view_as_real(sample)
        probe = torch.view_as_real(probe)
        sample_patches = torch.view_as_real(sample_patches)

        return noisy_data, scan_coords, sample, probe, sample_patches


def img2patch(img, coords):
    """
    Decomposes an image into patches based on the given coordinates.

    Args:
        img (torch.Tensor): The input image.
        coords (numpy.ndarray): An array containing the top-left and bottom-right coordinates of the patches.

    Returns:
        torch.Tensor: The decomposed image patches.
    """

    return torch.stack(
        [img[coords[i][0]:coords[i][1], coords[i][2]:coords[i][3]] for i in range(coords.shape[0])])


def fwd_patches_complex(obj_patches, probe):
    """
    Computes the complex diffraction pattern from image patches and a probe.

    Args:
        obj_patches (torch.Tensor): Input image patches.
        probe (torch.Tensor): Input probe.

    Returns:
        torch.Tensor: The complex diffraction patterns for each patch.
    """

    output = obj_patches * probe

    output = torch.fft.fftshift(output, dim=(-2, -1))
    output = torch.fft.fft2(output, norm='ortho')
    output = torch.fft.ifftshift(output, dim=(-2, -1))

    return output


def fwd_complex(obj, probe, coords):
    """
    Computes the complex diffraction pattern for an image using a probe.

    Args:
        obj (torch.Tensor): The input image.
        probe (torch.Tensor): The input probe.
        coords (numpy.ndarray): An array containing the top-left and bottom-right coordinates of the patches.

    Returns:
        torch.Tensor: The complex diffraction patterns for each patch.
    """

    obj_patches = img2patch(obj, coords)

    output = fwd_patches_complex(obj_patches, probe)

    return output
