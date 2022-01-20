import os
import importlib
import json
from pathlib import Path


# from dataset import lif_dataset as ldata
from network import utility
from utils import exp_util, vis_util
import system.ext
import time

import torch
import open3d as o3d
import numpy as np
from sklearn.neighbors import NearestNeighbors

import gc

def encoding(input_pc, encoder, voxel_size, device):
    input_pc_xyz = np.array(input_pc.points)
    input_pc_normal = np.array(input_pc.normals)
    input_pc_array = np.concatenate([input_pc_xyz, input_pc_normal], axis=1)

    # Split into local voxels
    print("Split to local voxels...")
    voxel_size = voxel_size
    bound_min = np.amin(input_pc_xyz, axis=0) - 0.2
    input_pc_xyz_zeroed = input_pc_xyz - bound_min
    input_pc_xyz_normalized = input_pc_xyz_zeroed / voxel_size
    voxel_locs = np.ceil(input_pc_xyz_normalized) - 1
    voxel_locs = np.unique(voxel_locs, axis=0)

    voxel_centers = bound_min + (voxel_locs + 0.5) * voxel_size
    nbrs_local = NearestNeighbors(radius=voxel_size, metric='chebyshev').fit(input_pc_xyz)
    local_indices = nbrs_local.radius_neighbors(voxel_centers, return_distance=False)

    # Normalize data and encoding
    print("Encoding...")
    voxel_locations = []
    latent_vects = []
    for vox_loc, vox_center, local_index in zip(voxel_locs, voxel_centers, local_indices):
        if len(local_index) == 0:
            continue
        vox_min = vox_center - 0.5 * voxel_size
        vox_max = vox_center + 0.5 * voxel_size
        vox_loc = torch.from_numpy(vox_loc).to(device).long()
        voxel_locations.append(vox_loc.unsqueeze(0))

        # Normalize pc
        input_pc_xyzn = input_pc_array[local_index]
        input_pc_xyzn[:, :3] = (input_pc_xyzn[:, :3] - vox_center) / (vox_max - vox_min)
        # Encoding
        input_pc_xyzn = torch.from_numpy(input_pc_xyzn).to(device).float()
        with torch.no_grad():
            latent_vect = encoder(input_pc_xyzn.unsqueeze(0))
        latent_vects.append(latent_vect)

    voxel_locations = torch.cat(voxel_locations, dim=0)
    latent_vects = torch.cat(latent_vects, dim=0)
    return bound_min, voxel_locations, latent_vects 


def completion(voxel_locations, latent_vects, completor, device):
    print("Do completion...")
    start = time.time()
    voxel_locations = torch.cat([voxel_locations, torch.zeros(voxel_locations.shape[0], 1, device=device)], dim=1)
    [voxel_locations, latent_vects], output_occ = completor([voxel_locations, latent_vects])
    voxel_locations = voxel_locations[:, 0:3]
    end = time.time()
    print(end-start)
    return voxel_locations, latent_vects 


def decoding(voxel_locations, latent_vects, decoder, voxel_resolution, voxel_size, bound_min, device):
    # Decoding
    print("Decoding...")
    max_volume = torch.amax(voxel_locations, dim=0) + 1
    max_volume = max_volume.cpu().numpy()
    indexer = torch.ones(np.prod(max_volume), device=device, dtype=torch.long) * -1
    voxel_locations_linear = voxel_locations[:, 2] + max_volume[-1] * voxel_locations[:, 1] + (max_volume[-1] * max_volume[-2]) * voxel_locations[:, 0]
    indexer[voxel_locations_linear] = torch.arange(0, latent_vects.shape[0], device=device, dtype=torch.long)

    B = latent_vects.shape[0]
    vec_id_batch_mapping = torch.arange(0, latent_vects.shape[0], device=device, dtype=torch.int)  

    # Sample data
    fast = False

    voxel_resolution = voxel_resolution
    sample_a = -(voxel_resolution // 2) * (1. / voxel_resolution)
    sample_b = 1. + (voxel_resolution - 1) // 2 * (1. / voxel_resolution)
    voxel_resolution *= 2

    relative_network_offset = torch.tensor([[0.5, 0.5, 0.5]], device=device, dtype=torch.float32)
    low_resolution = voxel_resolution // 2 if fast else voxel_resolution
    low_samples = utility.get_samples(low_resolution, device, a=sample_a, b=sample_b) - relative_network_offset # (l**3, 3)

    low_samples = low_samples.unsqueeze(0).repeat(B, 1, 1)  # (B, l**3, 3)
    latent_vects_pred = latent_vects.unsqueeze(1).repeat(1, low_samples.size(1), 1)  # (B, l**3, L)
    with torch.no_grad():
        low_sdf, low_std = utility.forward_model(decoder,
                                            latent_input=latent_vects_pred.view(-1, latent_vects_pred.size(-1)),
                                            xyz_input=low_samples.view(-1, low_samples.size(-1)), max_sample=100000)   

    if fast:
        low_sdf = low_sdf.reshape(B, 1, low_resolution, low_resolution, low_resolution)  # (B, 1, l, l, l)
        low_std = low_std.reshape(B, 1, low_resolution, low_resolution, low_resolution)
        high_sdf = torch.nn.functional.interpolate(low_sdf, mode='trilinear',
                                                    size=(voxel_resolution, voxel_resolution, voxel_resolution),
                                                    align_corners=True)
        high_std = torch.nn.functional.interpolate(low_std, mode='trilinear',
                                                    size=(voxel_resolution, voxel_resolution, voxel_resolution),
                                                    align_corners=True)
        high_sdf = high_sdf.squeeze(0).reshape(B, voxel_resolution ** 3)  # (B, H**3)
        high_std = high_std.squeeze(0).reshape(B, voxel_resolution ** 3)

        high_valid_lifs, high_valid_sbs = torch.where(high_sdf.abs() < 0.05)
        if high_valid_lifs.size(0) > 0:
            high_samples = utility.get_samples(voxel_resolution, device, a=sample_a, b=sample_b) - \
                            relative_network_offset  # (H**3, 3)
            high_latents = latent_vects[high_valid_lifs]  # (VH, 125)
            high_samples = high_samples[high_valid_sbs]  # (VH, 3)

            with torch.no_grad():
                # Decrease max_sample helps moderate memory problem in batch inference
                high_valid_sdf, high_valid_std = utility.forward_model(decoder,
                                                            latent_input=high_latents,
                                                            xyz_input=high_samples, max_sample=100000)
            high_sdf[high_valid_lifs, high_valid_sbs] = high_valid_sdf.squeeze(-1)
            high_std[high_valid_lifs, high_valid_sbs] = high_valid_std.squeeze(-1)

        high_sdf = high_sdf.reshape(B, voxel_resolution, voxel_resolution, voxel_resolution)
        high_std = high_std.reshape(B, voxel_resolution, voxel_resolution, voxel_resolution)
    else:
        high_sdf = low_sdf.reshape(B, low_resolution, low_resolution, low_resolution)
        high_std = low_std.reshape(B, low_resolution, low_resolution, low_resolution)
    
    print(f"5 Current GPU memory usage is {torch.cuda.memory_allocated(device=0) / 10**6}MB; Peak was {torch.cuda.max_memory_allocated(device=0) / 10**6}MB")
    del low_sdf, low_std
    gc.collect()
    torch.cuda.empty_cache()
    print(f"5 Current GPU memory usage is {torch.cuda.memory_allocated(device=0) / 10**6}MB; Peak was {torch.cuda.max_memory_allocated(device=0) / 10**6}MB")

    high_sdf = -high_sdf
    # In DI-Fusion, the non-interpolated marching cube is deprecated
    vertices, vertices_flatten_id, vertices_std = system.ext.marching_cubes_interp(
            indexer.view(max_volume.tolist()), voxel_locations_linear, vec_id_batch_mapping, high_sdf, high_std, int(1e7), max_volume.tolist(), 2000)  # (T, 3, 3), (T, ), (T, 3)

    vertices = vertices.cpu().numpy().reshape((-1, 3))
    vertices = vertices * voxel_size + bound_min
    triangles = np.arange(vertices.shape[0]).reshape((-1, 3))

    final_mesh = o3d.geometry.TriangleMesh()
    # The pre-conversion is saving tons of time
    final_mesh.vertices = o3d.utility.Vector3dVector(vertices.astype(float))
    final_mesh.triangles = o3d.utility.Vector3iVector(triangles.astype(np.int32))
    final_mesh.compute_vertex_normals()

    return final_mesh
    


def test(args, device, encoder, decoder, completor):
    print("Start testing...")

    if args.input_file[-3:] == "ply":
        args.do_encoding = True
        # Load input pc and convert into (N, 6) array
        input_pc = o3d.io.read_point_cloud(args.input_file)
        input_pc = input_pc.voxel_down_sample(voxel_size=0.005)
        if len(input_pc.normals) == 0:
            input_pc.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    elif args.input_file[-3:] == "npz":
        args.do_encoding = False
        data = np.load(args.input_file)
        bound_min = data["bound_min"]
        voxel_locations = torch.from_numpy(data["locs"]).to(device)
        latent_vects = torch.from_numpy(data["latent_vects"]).to(device)
        args.voxel_size = float(data["voxel_size"])


    # Encoding
    if args.do_encoding:
        bound_min, voxel_locations, latent_vects = encoding(input_pc, encoder, args.voxel_size, device)    

        if args.vis:
            # Create visualization
            voxel_centers = bound_min + (voxel_locations.detach().cpu().numpy() + 0.5) * args.voxel_size
            voxel_list = []
            for voxel_idx in range(len(voxel_centers)):
                vox = vis_util.wireframe_bbox(voxel_centers[voxel_idx, :] - 0.5*args.voxel_size, voxel_centers[voxel_idx, :] + 0.5*args.voxel_size, color_id=-1)
                voxel_list.append(vox)
            o3d.visualization.draw_geometries([input_pc]+voxel_list)
            o3d.visualization.draw_geometries(voxel_list)
    
    # Completion
    if args.do_completion:
        voxel_locations, latent_vects = completion(voxel_locations, latent_vects, completor, device)
    
        if args.vis:
            # Create visualization
            voxel_centers = bound_min + (voxel_locations.detach().cpu().numpy() + 0.5) * args.voxel_size
            voxel_list = []
            for voxel_idx in range(len(voxel_centers)):
                vox = vis_util.wireframe_bbox(voxel_centers[voxel_idx, :] - 0.5*args.voxel_size, voxel_centers[voxel_idx, :] + 0.5*args.voxel_size, color_id=2)
                voxel_list.append(vox)
            o3d.visualization.draw_geometries(voxel_list)
    
    # Decoding
    final_mesh = decoding(voxel_locations, latent_vects, decoder, args.resolution, args.voxel_size, bound_min, device)

    # Save mesh & visualize
    output_path = Path(args.output_file).parent
    output_path.mkdir(parents=True, exist_ok=True)
    o3d.io.write_triangle_mesh(args.output_file, final_mesh)
    if args.vis:
        voxel_centers = bound_min + (voxel_locations.detach().cpu().numpy() + 0.5) * args.voxel_size
        voxel_list = []
        for voxel_idx in range(len(voxel_centers)):
            vox = vis_util.wireframe_bbox(voxel_centers[voxel_idx, :] - 0.5*args.voxel_size, voxel_centers[voxel_idx, :] + 0.5*args.voxel_size)
            voxel_list.append(vox)
        o3d.visualization.draw_geometries([final_mesh]+voxel_list)
    


if __name__ == '__main__':

    parser = exp_util.ArgumentParserX(add_hyper_arg=True)
    parser.add_argument('-v', '--vis', action='store_true', help='Visualize')
    parser.add_argument('-c', '--do_completion', action='store_true', help='Do completion')

    args = parser.parse_args()

    # Load encoder and decoder
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model, args_model = utility.load_model("ckpt/default/hyper.json", 300, mode='train')  # models already on cuda
    model, args_model = utility.load_model(args.training_hyper_path, args.use_epoch, mode='train')  # models already on cuda
    model.eval()

    # Load completor
    training_hyper_path = Path(args.training_hyper_path)
    completor_args = exp_util.parse_config_json(training_hyper_path)
    exp_dir = training_hyper_path.parent
    model_paths = exp_dir.glob('completor_*.pth.tar')
    model_paths = {int(str(t).split("completor_")[-1].split(".pth")[0]): t for t in model_paths}
    assert args.use_epoch in model_paths.keys(), f"{args.use_epoch} not found in {sorted(list(model_paths.keys()))}"
    args.checkpoint = model_paths[args.use_epoch]
    print("Load checkpoint: ", args.checkpoint)

    completor_module = importlib.import_module("network." + completor_args.completor_name)
    completor = completor_module.Model(completor_args.code_length, **completor_args.completor_specs).to(device)
    state_dict = torch.load(args.checkpoint)["model_state"]
    completor.load_state_dict(state_dict)
    completor.eval()

    test(args, device, model.encoder, model.decoder, completor)




