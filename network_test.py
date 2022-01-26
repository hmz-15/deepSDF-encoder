from asyncio.format_helpers import _format_callback_source
import os
import importlib
import json
from pathlib import Path
import glob

from utils import exp_util, vis_util
import time

import torch
import open3d as o3d
import numpy as np
import skimage.measure as measure


def load_models(training_hyper_path, use_epoch, device):
    training_hyper_path = Path(training_hyper_path)
    args = exp_util.parse_config_json(training_hyper_path)
    exp_dir = training_hyper_path.parent
    model_paths = exp_dir.glob('decoder_*.pth.tar')
    model_paths = {int(str(t).split("decoder_")[-1].split(".pth")[0]): t for t in model_paths}
    assert use_epoch in model_paths.keys(), f"{use_epoch} not found in {sorted(list(model_paths.keys()))}"
    args.checkpoint = model_paths[use_epoch]
    print("Load checkpoint: ", args.checkpoint)

    # Load models
    decoder_module = importlib.import_module("network." + args.decoder_name)
    decoder = decoder_module.Model(args.code_length, **args.decoder_specs).to(device)
    encoder_module = importlib.import_module("network." + args.encoder_name)
    encoder = encoder_module.Model(**args.encoder_specs, mode='train').to(device)

    # Load checkpoint
    if args.checkpoint is not None:
        state_dict = torch.load(args.checkpoint)["model_state"]
        decoder.load_state_dict(state_dict)
        state_dict = torch.load(Path(args.checkpoint).parent / f"encoder_{use_epoch}.pth.tar")["model_state"]
        encoder.load_state_dict(state_dict)    

    print("Number of decoder parameters: {}".format(sum(p.data.nelement() for p in decoder.parameters())))
    print("Number of encoder parameters: {}".format(sum(p.data.nelement() for p in encoder.parameters())))

    return encoder, decoder    


def test_single(args, device, encoder, decoder, input_pc, gt_sdf_data=None):
    # Normalize
    input_pc_xyz = np.array(input_pc.points)
    input_pc_normal = np.array(input_pc.normals)
    offset = (np.amax(input_pc_xyz, axis=0) + np.amin(input_pc_xyz, axis=0)) / 2
    scale = np.linalg.norm(np.amax(input_pc_xyz, axis=0) - np.amin(input_pc_xyz, axis=0))

    input_pc_xyz = (input_pc_xyz - offset) / scale
    if not gt_sdf_data is None:
        gt_sdf_data /= scale

    # Encoding
    print("Encoding...")
    
    input_pc_array = np.concatenate([input_pc_xyz, input_pc_normal], axis=1)
    input_pc_tensor = torch.from_numpy(input_pc_array).to(device).float()

    with torch.no_grad():
        latent_vect = encoder(input_pc_tensor.unsqueeze(0))    

    # Decoding 1 (generate mesh)
    print("Decoding for mesh generation...")
    # The voxel_origin is actually the (bottom, left, down) corner, not the middle
    N = args.resolution
    voxel_origin = [-1, -1, -1]
    voxel_size = 2.0 / (N - 1)

    overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor())
    samples = torch.zeros(N ** 3, 4, device=device)

    # transform first 3 columns
    # to be the x, y, z index
    samples[:, 2] = overall_index % N
    samples[:, 1] = (overall_index.long() / N) % N
    samples[:, 0] = ((overall_index.long() / N) / N) % N

    # transform first 3 columns
    # to be the x, y, z coordinate
    samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
    samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
    samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]
    num_samples = N ** 3

    # Feed into decoder
    latent_repeat = latent_vect.expand(num_samples, -1)
    # net_input = torch.cat([latent_repeat, samples[:,0:3]], dim=1)
    # with torch.no_grad():
    #     sdf_values = decoder(net_input).data.cpu().numpy()
    # sdf_values = sdf_values.reshape(N, N, N)

    batch_split = int(num_samples / args.max_samples_per_split)
    xyz_chunk = torch.chunk(samples[:,0:3], batch_split)
    lat_vecs_chunk = torch.chunk(latent_repeat, batch_split)

    sdf_list = []
    for i in range(batch_split):
        net_input = torch.cat([lat_vecs_chunk[i], xyz_chunk[i]], dim=1)
        with torch.no_grad():
            sdf_values = decoder(net_input)
        sdf_list.append(sdf_values)
    
    sdf_values = torch.cat([sdf.reshape(-1, 1) for sdf in sdf_list], dim=0).data.cpu().numpy()
    sdf_values = sdf_values.reshape(N, N, N)

    # Generate mesh
    vertices, triangles, normals, values = measure.marching_cubes(
        sdf_values, level = 0.0, spacing=[voxel_size] * 3)
    
    # vertices[:, [0,1]] = vertices[:, [1,0]]
    vertices = vertices + voxel_origin
    vertices = vertices * scale + offset

    final_mesh = o3d.geometry.TriangleMesh()
    # The pre-conversion is saving tons of time
    final_mesh.vertices = o3d.utility.Vector3dVector(vertices.astype(float))
    final_mesh.triangles = o3d.utility.Vector3iVector(triangles.astype(np.int32))
    final_mesh.compute_vertex_normals()

    # Decoding 2 (evaluate sdf)
    if gt_sdf_data is not None:
        print("Decoding for sdf evaluation...")
        gt_sdf_data = torch.from_numpy(gt_sdf_data).to(device)
        latent_repeat = latent_vect.expand(gt_sdf_data.shape[0], -1)
        net_input = torch.cat([latent_repeat, gt_sdf_data[:,0:3]], dim=1)
        sdf_values = decoder(net_input)
        # .data.cpu().numpy()
        print((sdf_values.squeeze() - gt_sdf_data[:,3]).abs().sum())

    return final_mesh, latent_vect.cpu().numpy()[0,:], scale


if __name__ == '__main__':

    parser = exp_util.ArgumentParserX(add_hyper_arg=True)
    parser.add_argument('-v', '--vis', action='store_true', help='Visualize')

    args = parser.parse_args()

    # Load encoder and decoder
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder, decoder = load_models(args.training_hyper_path, args.use_epoch, device)  # models already on cuda
    encoder.eval()
    decoder.eval()

    # Load input data
    gt_file_list = None
    if os.path.isdir(args.input_path):  
        file_list = sorted(glob.glob(args.input_path + "/*.ply"))
        try:
            gt_file_path = str(Path(args.input_path).parent / "payload")
            gt_file_list = sorted(glob.glob(gt_file_path + "/*.npz"))
        except:
            pass
    elif os.path.isfile(args.input_path):
        assert args.input_path.split(".")[-1] == "ply", "Input has to be in the .ply format"
        file_list = [args.input_path]

    # Test
    print("Start testing...")
    latent_dict = dict()
    latent_dim = None
    counter = 0
    for file in file_list:
        # Load input pc and convert into (N, 6) array
        input_pc = o3d.io.read_point_cloud(file)
        if len(input_pc.normals) == 0:
            input_pc.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

        # Load ground truth sdf if available
        gt_sdf_data = None
        if gt_file_list is not None:
            data = np.load(gt_file_list[counter])
            gt_sdf_data = data["sdf_data"]

        # Create visualization
        # if args.vis:
        #     o3d.visualization.draw_geometries([input_pc])

        # Test for single pointcloud
        output_mesh, latent_vect, scale = test_single(args, device, encoder, decoder, input_pc, gt_sdf_data)
        print("latent vector: ", latent_vect)
        print("scale: ", scale)

        latent_dim = len(latent_vect)

        # Save mesh & visualize
        output_path = Path(args.output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        output_file = str(output_path) + "/reconstructed_" + file.split("/")[-1]
        o3d.io.write_triangle_mesh(output_file, output_mesh)

        object_id = file.split('/')[-1].split(".")[0]
        latent_dict[object_id] = latent_vect

        if args.vis:
            o3d.visualization.draw_geometries([output_mesh, input_pc])
        
        counter += 1
        if (counter >= args.max_shape_num):
            break

    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    latent_file = str(output_path) + "/latent_vectors.csv"

    with open(latent_file, "w") as fout:
        format_str = "{}," * (latent_dim + 1)
        format_str = format_str[:-1] + '\n'
        for k, v in latent_dict.items():
            fout.write(format_str.format(k, *v))
    
    print("latent vectors are saved at: {}".format(latent_file))
