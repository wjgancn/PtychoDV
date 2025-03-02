import os
import yaml
import pytorch_lightning as pl

from ptycho_dv import vit
from ptycho_dv.dataset import WaveProDataset
import torch
import matplotlib.pyplot as plt
import numpy as np
import time

# Load the pre-defined configuration file.
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Instantiate the testing dataset class.
testing_dataset = WaveProDataset(mode='tst', config=config)

# Instantiate the model and load the weights from the checkpoint file.
model = vit.PtyChoModule.load_from_checkpoint(
        checkpoint_path=os.path.join(config['test']['checkpoint_path']),
        config=config,
        map_location='cpu'
)
model = model.cpu()

# Load data from the testing dataset.
noisy_data, scan_coords, sample, probe, sample_patches = testing_dataset[0]

noisy_data = noisy_data.squeeze(-1).numpy()
sample = torch.view_as_complex(sample).numpy()
probe = torch.view_as_complex(probe).numpy()
sample_patches = torch.view_as_complex(sample_patches).numpy()

# Preprocess the data for further computation.
noisy_data_cpu = torch.from_numpy(noisy_data).unsqueeze(-1).unsqueeze(0)
scan_coords_cpu = scan_coords.unsqueeze(0)
probe_cpu = torch.view_as_real(torch.from_numpy(probe)).unsqueeze(0)

# Run the model on the input data to obtain reconstructed images.
with torch.no_grad():
    model(noisy_data_cpu, scan_coords_cpu, probe_cpu, noisy_data_cpu)  # a dummy run to get an accurate running time.

    print("Start reconstruction of PtychoViT ...")    
    time_init = time.time()
    _, ptychoViT_obj, _ = model(noisy_data_cpu, scan_coords_cpu, probe_cpu, noisy_data_cpu)

# Reformat the reconstructed results and visualize them using matplotlib.
ptychoViT_obj = ptychoViT_obj.squeeze(0)
ptychoViT_obj = ptychoViT_obj[..., 0] + ptychoViT_obj[..., 1] * 1j
ptychoViT_obj = ptychoViT_obj.cpu().numpy()

print("Visualizing the results ...")
plt.figure(figsize=[6, 7])

CROP_COORDINATE = (128, 128, 256, 256) 
sample = sample[CROP_COORDINATE[0]:CROP_COORDINATE[2], CROP_COORDINATE[1]:CROP_COORDINATE[3]]
ptychoViT_obj = ptychoViT_obj[CROP_COORDINATE[0]:CROP_COORDINATE[2], CROP_COORDINATE[1]:CROP_COORDINATE[3]]

plt.subplot(2, 2, 1)
plt.imshow(abs(sample), cmap='gray', vmin=np.amin(abs(sample)), vmax=np.amax(abs(sample)))
plt.axis('off')
plt.title('(Magnitude) Ground-truth')

plt.subplot(2, 2, 2)
plt.imshow(abs(ptychoViT_obj), cmap='gray', vmin=np.amin(abs(sample)), vmax=np.amax(abs(sample)))
plt.axis('off')
plt.title('(Magnitude) PtychoDV')

plt.subplot(2, 2, 3)
plt.imshow(np.angle(sample), cmap='gray', vmin=np.amin(np.angle(sample)), vmax=np.amax(np.angle(sample)))
plt.axis('off')
plt.title('(Angle) Ground-truth')

plt.subplot(2, 2, 4)
plt.imshow(np.angle(ptychoViT_obj), cmap='gray', vmin=np.amin(np.angle(sample)), vmax=np.amax(np.angle(sample)))
plt.axis('off')
plt.title('(Angle) PtychoDV')

plt.tight_layout()

plt.show()
