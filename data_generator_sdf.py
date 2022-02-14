import numpy as np
import functools
from multiprocessing import Pool, Value, Manager
from sklearn.neighbors import NearestNeighbors
from pathlib import Path
import random

from utils import exp_util, vis_util
from pyquaternion import Quaternion
import open3d as o3d

import subprocess
import json
import shutil
import os
import argparse
import logging


CUDA_SAMPLER_PATH = Path(__file__).resolve().parent / "sampler_cuda" / "bin" / "PreprocessMeshCUDA"
_counter = Value('i', 0)
_bad_counter = Value('i', 0)


def generate_samples(idx: int, args: argparse.ArgumentParser, provider, output_base, source_list, vis: bool = False):
    mesh_path, vcam, ref_bin_path, sampler_mult = provider[idx]
    raw_obj_mesh = o3d.io.read_triangle_mesh(mesh_path)
    raw_mesh_verts = np.asarray(raw_obj_mesh.vertices)
    shape_size = np.linalg.norm(np.amax(raw_mesh_verts, axis=0) - np.amin(raw_mesh_verts, axis=0))
    offset = (np.amax(raw_mesh_verts, axis=0) + np.amin(raw_mesh_verts, axis=0))/2

    raw_mesh_verts = (raw_mesh_verts - offset) / shape_size
    sampler_mult /= shape_size

    print(mesh_path)

    def call_cuda_sampler(num_vcam):
        # Tmp file for sampling.
        output_tmp_path = output_base / ("%06d.raw" % idx)
        surface_tmp_path = output_base / ("%06d.surf" % idx)
        vcam_file_path = output_base / ("%06d.cam" % idx)

        # Sample cameras
        vcam_ext_sampled = np.random.choice(vcam[1], num_vcam, replace=False)
        # Save the camera
        with vcam_file_path.open('wb') as f:
            np.asarray(vcam[0]).flatten().astype(np.float32).tofile(f)
            np.asarray([cam.to_gl_camera().inv().matrix.T for cam in vcam_ext_sampled]).reshape(-1, 16).astype(np.float32).tofile(f)
        # Call CUDA sampler
        arg_list_common = [str(CUDA_SAMPLER_PATH),
                        '-m', mesh_path,
                        '-s', str(int(args.sampler_count)),
                        '-o', str(output_tmp_path),
                        '-c', str(vcam_file_path),
                        '-r', str(args.sample_method),
                        '--surface', str(surface_tmp_path)]
        arg_list_data = arg_list_common + ['-p', '0.8', '--var', str(args.sampler_var), '-e', str(0.2)]
        if ref_bin_path is not None:
            arg_list_data += ['--ref', ref_bin_path, '--max_ref_dist', str(args.max_ref_dist)]

        is_bad = False
        sampler_pass = args.__dict__.get("sampler_pass", 1)

        data_arr = []
        surface_arr = []
        for sid in range(sampler_pass):
            subproc = subprocess.Popen(arg_list_data, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            subproc.wait()
            # Read raw file and convert it to numpy file.
            try:
                cur_data_arr = np.fromfile(str(output_tmp_path), dtype=np.float32).reshape(-1, 4)
                cur_surface_arr = np.fromfile(str(surface_tmp_path), dtype=np.float32).reshape(-1, 6)
                data_arr.append(cur_data_arr)
                surface_arr.append(cur_surface_arr)
                os.unlink(output_tmp_path)
                os.unlink(surface_tmp_path)
            except FileNotFoundError:
                print(' '.join(arg_list_data))
                is_bad = True
                break
        # Do cleaning of sampler.
        os.unlink(vcam_file_path)
        return is_bad, data_arr, surface_arr

    # Test watertight
    if not raw_obj_mesh.is_watertight():
        is_bad = True
    else:
        # Sample full data
        is_bad, data_arr, surface_arr = call_cuda_sampler(len(vcam[1]))

    if is_bad:
        print("Bad!")
        provider.clean(idx)
        with _bad_counter.get_lock():
            _bad_counter.value += 1
        return

    data_arr = np.concatenate(data_arr, axis=0)
    data_arr[:,0:3] -= offset
    data_arr *= sampler_mult
    surface_arr = np.concatenate(surface_arr, axis=0)
    surface_arr[:,0:3] -= offset
    surface_arr[:,0:3] *= sampler_mult

    # Badly, some surface arr may have NaN normal, we prune them
    surface_arr_nan_row = np.any(np.isnan(surface_arr), axis=1)
    surface_arr = surface_arr[~surface_arr_nan_row]

    if vis:
        # Create visualization
        surface_pcd = o3d.geometry.PointCloud()
        surface_pcd.points = o3d.utility.Vector3dVector(surface_arr[:, :3])
        surface_pcd.normals = o3d.utility.Vector3dVector(surface_arr[:, 3:])
        o3d.visualization.draw_geometries([surface_pcd])

        sdf_pcd = o3d.geometry.PointCloud()
        sdf_pcd.points = o3d.utility.Vector3dVector(data_arr[:, :3])
        o3d.visualization.draw_geometries([sdf_pcd])
    
    # Save data
    with _counter.get_lock():
        mesh_idx = _counter.value
        _counter.value += 1
        source_list.append([provider.get_source(idx), mesh_idx])
        print(f"{_counter.value}/ {len(provider)}, bad shape {_bad_counter.value}/ {len(provider)}")

    output_data = {"sdf_data": data_arr,
                   "pc_data": surface_arr}
    output_data_base = output_base / "payload"
    np.savez(output_data_base / ("%06d.npz" % mesh_idx), **output_data)

    # Output pc
    if args.write_pc:
        print("Output pc...")
        output_pc_path = output_base / "pc" / ("%06d.ply" % mesh_idx)
        surface_pcd = o3d.geometry.PointCloud()
        surface_pcd.points = o3d.utility.Vector3dVector(surface_arr[:, :3])
        surface_pcd.normals = o3d.utility.Vector3dVector(surface_arr[:, 3:])
        o3d.io.write_point_cloud(str(output_pc_path), surface_pcd)   

    # Output mesh
    if args.write_normalized_mesh:
        print("Output mesh...")
        output_mesh_path = output_base / "mesh" / ("%06d.obj" % mesh_idx)
        mesh_norm = o3d.geometry.TriangleMesh()
        mesh_norm.vertices = o3d.utility.Vector3dVector(raw_mesh_verts)
        mesh_norm.triangles = raw_obj_mesh.triangles
        o3d.io.write_triangle_mesh(str(output_mesh_path), mesh_norm)


if __name__ == '__main__':
    from dataset.cutting_shape import CuttingShape
    logging.basicConfig(level=logging.INFO)

    exp_util.init_seed(4)
    mesh_providers = {
        'cutting_shape': CuttingShape,
        # 'shapenet_model': ShapeNetGenerator,
    }

    parser = exp_util.ArgumentParserX(add_hyper_arg=True, description='SDF Data Generator.')
    parser.add_argument('-v', '--visualize', action='store_true', help='Visualize')
    args = parser.parse_args()

    print(args)

    dataset = mesh_providers[args.provider](**args.provider_kwargs)
    output_path = Path(args.output)

    if output_path.exists():
        print("Removing old dataset...")
        shutil.rmtree(output_path)

    output_path.mkdir(parents=True, exist_ok=True)

    (output_path / "pc").mkdir(exist_ok=True, parents=True)
    (output_path / "mesh").mkdir(exist_ok=True, parents=True)
    (output_path / "payload").mkdir(exist_ok=True, parents=True)

    with (output_path / "config.json").open("w") as f:
        json.dump(vars(args), f, indent=2)

    # Shared structures:
    manager = Manager()
    source_list = manager.list()

    if args.nproc > 0:
        p = Pool(processes=args.nproc)
        p.map(functools.partial(generate_samples, args=args, output_base=output_path,
                                provider=dataset, source_list=source_list, vis=args.visualize), range(len(dataset)))
    else:
        for idx in range(len(dataset)):
            generate_samples(idx, args, dataset, output_path, source_list, vis=args.visualize)

    with (output_path / "source.json").open("w") as f:
        json.dump(list(source_list), f, indent=2)

    print(f"Done with {_bad_counter.value} bad shapes")
