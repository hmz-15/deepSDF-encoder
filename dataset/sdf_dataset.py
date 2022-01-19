from pathlib import Path
import torch
from torch.utils import data
import json
import numpy as np
import time


def perturb_normal(normals, theta_range):
    normal_x_1 = np.stack([-normals[:, 1], normals[:, 0], np.zeros_like(normals[:, 0])], axis=1)
    normal_x_2 = np.stack([-normals[:, 2], np.zeros_like(normals[:, 0]), normals[:, 0]], axis=1)
    normal_x_mask = np.abs(np.abs(normals[:, 2]) - 1.0) > 0.1
    normal_x = np.zeros_like(normals)
    normal_x[normal_x_mask] = normal_x_1[normal_x_mask]
    normal_x[~normal_x_mask] = normal_x_2[~normal_x_mask]
    normal_x /= np.linalg.norm(normal_x, axis=1, keepdims=True)
    normal_y = np.cross(normals, normal_x)

    phi = np.random.rand(normal_x.shape[0], 1) * 2.0 * np.pi
    phi_dir = np.cos(phi) * normal_x + np.sin(phi) * normal_y
    theta = np.random.rand(normal_x.shape[0], 1) * theta_range
    perturbed_normal = np.cos(theta) * normals + np.sin(theta) * phi_dir
    return perturbed_normal


# def collate_full(batch):
#     # locs.float(), lif_samples.float(), full_pc_samples.float(), corr_locs.float(), corr_pc_samples.float()
#     batch = list(filter(lambda x: len(x[0])!=0, batch))
#     if len(batch) == 0:
#         return [], -1
#     idx = [batch[0][1]]
#     lif_samples = batch[0][0][1]
#     full_pc_samples = batch[0][0][2]
#     corr_pc_samples = batch[0][0][4]

#     locs = batch[0][0][0]
#     locs = torch.cat([locs, torch.zeros(locs.shape[0], 1).long()], dim=1)
#     corr_locs = batch[0][0][3]
#     corr_locs = torch.cat([corr_locs, torch.zeros(corr_locs.shape[0], 1).long()], dim=1)

#     precompute_occ_flag = False
#     if len(batch[0][0]) > 5:
#         precompute_occ_flag = True
#         occ_gt = batch[0][0][5]
#         occ_weights = batch[0][0][6]
    
#     for b in range(1, len(batch)):
#         idx += [batch[b][1]]
#         lif_samples = torch.cat([lif_samples, batch[b][0][1]], dim=0)
#         full_pc_samples = torch.cat([full_pc_samples, batch[b][0][2]], dim=0)
#         corr_pc_samples = torch.cat([corr_pc_samples, batch[b][0][4]], dim=0)

#         cur_locs = batch[b][0][0]
#         cur_locs = torch.cat([cur_locs, torch.ones(cur_locs.shape[0], 1).long()*b], dim=1)
#         locs = torch.cat([locs, cur_locs], dim=0)
#         cur_corr_locs = batch[b][0][3]
#         cur_corr_locs = torch.cat([cur_corr_locs, torch.ones(cur_corr_locs.shape[0], 1).long()*b], dim=1)
#         corr_locs = torch.cat([corr_locs, cur_corr_locs], dim=0)

#         if precompute_occ_flag:
#             occ_gt = torch.cat([occ_gt, batch[b][0][5]], dim=0)
#             occ_weights = torch.cat([occ_weights, batch[b][0][6]], dim=0)

#     if precompute_occ_flag:
#         return [locs.long(), lif_samples.float(), full_pc_samples.float(), corr_locs.long(), corr_pc_samples.float(), occ_gt, occ_weights], idx
#     else:
#         return [locs.long(), lif_samples.float(), full_pc_samples.float(), corr_locs.long(), corr_pc_samples.float()], idx


class SDFDataset(data.Dataset):
    def __init__(self, data_path, num_sample, num_surface_sample: int = 0, augment_noise=(0.0, 0.0)):
        self.data_path = Path(data_path)
        with (self.data_path / "source.json").open() as f:
            self.data_sources = json.load(f)
        self.num_sample = num_sample
        self.num_surface_sample = num_surface_sample
        self.surface_format = None
        self.augment_noise = augment_noise

    def __len__(self):
        return len(self.data_sources)

    def get_raw_data(self, idx):
        sdf_path = self.data_path / "payload" / ("%06d.npz" % idx)
        return np.load(sdf_path)

    def __getitem__(self, idx):
        if idx < 0:
            assert -idx <= len(self)
            idx = len(self) + idx

        # Load data
        try:
            data = self.get_raw_data(idx)
            sdf_data = data["sdf_data"]
            pc_data = data["pc_data"]
        except:
            return [None, idx]

        sdf_data = torch.from_numpy(sdf_data)
        pos_mask = sdf_data[:, 3] > 0
        pos_tensor = sdf_data[pos_mask]
        neg_tensor = sdf_data[torch.logical_not(pos_mask)]
        half = int(self.num_sample / 2)
        random_pos = (torch.rand(half) * pos_tensor.shape[0]).long()
        random_neg = (torch.rand(half) * neg_tensor.shape[0]).long()
        sample_pos = torch.index_select(pos_tensor, 0, random_pos)
        sample_neg = torch.index_select(neg_tensor, 0, random_neg)
        sdf_samples = torch.cat([sample_pos, sample_neg], dim=0)

        pc_data = pc_data[np.random.choice(pc_data.shape[0], size=self.num_surface_sample, replace=True), :]

        if self.augment_noise[0] > 0.0:
            pc_data[:, :3] += np.random.randn(pc_data.shape[0], 3) * self.augment_noise[0]
            pc_data[:, 3:6] = perturb_normal(pc_data[:, 3:6], np.deg2rad(self.augment_noise[1]))

        if not isinstance(pc_data, torch.Tensor):
            pc_data = torch.from_numpy(pc_data)            

        return [sdf_samples.float(), pc_data.float(), idx]


class SDFCombinedDataset(data.Dataset):
    def __init__(self, *datasets):
        super().__init__()
        assert len(datasets) > 0, 'datasets should not be an empty iterable'
        self.datasets = datasets
        self.cumulative_sizes = np.cumsum([len(ds) for ds in self.datasets])

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        if idx < 0:
            assert -idx <= len(self)
            idx = len(self) + idx
        dataset_idx = np.searchsorted(self.cumulative_sizes, idx, side='right')
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        dat = self.datasets[dataset_idx][sample_idx]
        dat = dat[:-1]

        if dat[0] is None:
            return [], idx

        return dat, idx
