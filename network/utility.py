import math
import torch
import torch.nn as nn
import logging
from utils import exp_util
from pathlib import Path
import importlib

# import sparseconvnet as scn


class Networks:
    def __init__(self):
        self.decoder = None
        self.encoder = None

    def eval(self):
        if self.encoder is not None:
            self.encoder.eval()
        if self.decoder is not None:
            self.decoder.eval()


def load_model(training_hyper_path: str, use_epoch: int = -1, mode: str = 'cnp'):
    """
    Load in the model and hypers used.
    :param training_hyper_path:
    :param use_epoch: if -1, will load the latest model.
    :return: Networks
    """
    training_hyper_path = Path(training_hyper_path)

    if training_hyper_path.name.split(".")[-1] == "json":
        args = exp_util.parse_config_json(training_hyper_path)
        exp_dir = training_hyper_path.parent
        model_paths = exp_dir.glob('decoder_*.pth.tar')
        model_paths = {int(str(t).split("decoder_")[-1].split(".pth")[0]): t for t in model_paths}
        assert use_epoch in model_paths.keys(), f"{use_epoch} not found in {sorted(list(model_paths.keys()))}"
        args.checkpoint = model_paths[use_epoch]
        print("Load checkpoint: ", args.checkpoint)
        
    else:
        args = exp_util.parse_config_yaml(Path('configs/training_defaults.yaml'))
        args = exp_util.parse_config_yaml(training_hyper_path, args)
        logging.warning("Loaded a un-initialized model.")
        args.checkpoint = None

    model = Networks()
    decoder_module = importlib.import_module("network." + args.decoder_name)
    model.decoder = decoder_module.Model(args.code_length, neighbor_mode=args.neighbor_mode, **args.decoder_specs).cuda()
    encoder_module = importlib.import_module("network." + args.encoder_name)
    model.encoder = encoder_module.Model(**args.encoder_specs, mode=mode).cuda()
    if args.checkpoint is not None:
        if model.decoder is not None:
            state_dict = torch.load(args.checkpoint)["model_state"]
            model.decoder.load_state_dict(state_dict)
        if model.encoder is not None:
            state_dict = torch.load(Path(args.checkpoint).parent / f"encoder_{use_epoch}.pth.tar")["model_state"]
            model.encoder.load_state_dict(state_dict)

    print("Number of decoder parameters: {}".format(sum(p.data.nelement() for p in model.decoder.parameters())))
    print("Number of encoder parameters: {}".format(sum(p.data.nelement() for p in model.encoder.parameters())))

    args.run_cat_neighbor = False if args.neighbor_mode is None else True

    return model, args


def forward_model(model: nn.Module, 
                  latent_input: torch.Tensor = None,
                  xyz_input: torch.Tensor = None,
                  loss_func=None, max_sample: int = 2 ** 32,
                  no_detach: bool = False,
                  verbose: bool = False):
    """
    Forward the neural network model. (if loss_func is not None, will also compute the gradient w.r.t. the loss)
    Either network_input or (latent_input, xyz_input) tuple could be provided.
    :param model:           MLP model.
    :param network_input:   (N, 128)
    :param latent_input:    (N, 125)
    :param xyz_input:       (N, 3)
    :param loss_func:
    :param max_sample
    :return: [(N, X)] several values
    """
    assert latent_input is not None and xyz_input is not None
        # assert network_input is None
        # network_input = torch.cat((latent_input, xyz_input), dim=1)

    # assert network_input.ndimension() == 2

    n_chunks = math.ceil(latent_input.size(0) / max_sample)
    assert not no_detach or n_chunks == 1

    latent_input = torch.chunk(latent_input, n_chunks)
    xyz_input = torch.chunk(xyz_input, n_chunks)

    if verbose:
        logging.debug(f"Network input chunks = {n_chunks}, each chunk = {latent_input[0].size()}")

    head = 0
    output_chunks = None
    for chunk_i, (latent_input_chunk, xyz_input_chunk) in enumerate(zip(latent_input, xyz_input)):
        # (N, 1)
        input_chunk = torch.cat((latent_input_chunk, xyz_input_chunk), dim=1)
        network_output = model(input_chunk)
        if not isinstance(network_output, tuple):
            network_output = [network_output, ]

        if chunk_i == 0:
            output_chunks = [[] for _ in range(len(network_output))]

        if loss_func is not None:
            # The 'graph' in pytorch stores how the final variable is computed to its current form.
            # Under normal situations, we can delete this path right after the gradient is computed because the path
            #   will be re-constructed on next forward call.
            # However, in our case, self.latent_vec is the leaf node requesting the gradient, the specific computation:
            #   vec = self.latent_vec[inds] && cat(vec, xyz)
            #   will be forgotten, too. if we delete the entire graph.
            # Indeed, the above computation is the ONLY part that we do not re-build during the next forwarding.
            # So, we set retain_graph to True.
            # According to https://github.com/pytorch/pytorch/issues/31185, if we delete the head loss immediately
            #   after the backward(retain_graph=True), the un-referenced part graph will be deleted too,
            #   hence keeping only the needed part (a sub-graph). Perfect :)
            loss_func(network_output,
                      torch.arange(head, head + network_output[0].size(0), device=network_output[0].device)
                      ).backward(retain_graph=(chunk_i != n_chunks - 1))
        if not no_detach:
            network_output = [t.detach() for t in network_output]

        for payload_i, payload in enumerate(network_output):
            output_chunks[payload_i].append(payload)
        head += network_output[0].size(0)

    output_chunks = [torch.cat(t, dim=0) for t in output_chunks]
    return output_chunks


def get_samples(r: int, device: torch.device, a: float = 0.0, b: float = None):
    """
    Get samples within a cube, the voxel size is (b-a)/(r-1). range is from [a, b]
    :param r: num samples
    :param a: bound min
    :param b: bound max
    :return: (r*r*r, 3)
    """
    overall_index = torch.arange(0, r ** 3, 1, device=device, dtype=torch.long)
    r = int(r)

    if b is None:
        b = 1. - 1. / r

    vsize = (b - a) / (r - 1)
    samples = torch.zeros(r ** 3, 3, device=device, dtype=torch.float32)
    samples[:, 0] = (overall_index // (r * r)) * vsize + a
    samples[:, 1] = ((overall_index // r) % r) * vsize + a
    samples[:, 2] = (overall_index % r) * vsize + a

    return samples


def pack_samples(sample_indexer: torch.Tensor, count: int,
                 sample_values: torch.Tensor = None):
    """
    Pack a set of samples into batches. Each element in the batch is a random subsampling of the sample_values
    :param sample_indexer: (N, )
    :param count: C
    :param sample_values: (N, L), if None, will return packed_inds instead of packed.
    :return: packed (B, C, L) or packed_inds (B, C), mapping: (B, ).
    """
    from system.ext import pack_batch

    # First shuffle the samples to avoid biased samples.
    shuffle_inds = torch.randperm(sample_indexer.size(0), device=sample_indexer.device)
    sample_indexer = sample_indexer[shuffle_inds]

    mapping, pinds, pcount = torch.unique(sample_indexer, return_inverse=True, return_counts=True)

    n_batch = mapping.size(0)
    packed_inds = pack_batch(pinds, n_batch, count * 2)         # (B, 2C)

    pcount.clamp_(max=count * 2 - 1)
    packed_inds_ind = torch.floor(torch.rand((n_batch, count), device=pcount.device) * pcount.unsqueeze(-1)).long()  # (B, C)

    packed_inds = torch.gather(packed_inds, 1, packed_inds_ind)     # (B, C)
    packed_inds = shuffle_inds[packed_inds]                         # (B, C)

    if sample_values is not None:
        assert sample_values.size(0) == sample_indexer.size(0)
        packed = torch.index_select(sample_values, 0, packed_inds.view(-1)).view(n_batch, count, sample_values.size(-1))
        return packed, mapping
    else:
        return packed_inds, mapping


def groupby_reduce(sample_indexer: torch.Tensor, sample_values: torch.Tensor, op: str = "max"):
    """
    Group-By and Reduce sample_values according to their indices, the reduction operation is defined in `op`.
    :param sample_indexer: (N,). An index, must start from 0 and go to the (max-1), can be obtained using torch.unique.
    :param sample_values: (N, L)
    :param op: have to be in 'max', 'mean'
    :return: reduced values: (C, L)
    """
    C = sample_indexer.max() + 1
    n_samples = sample_indexer.size(0)

    assert n_samples == sample_values.size(0), "Indexer and Values must agree on sample count!"

    if op == 'mean':
        from system.ext import groupby_sum
        values_sum, values_count = groupby_sum(sample_values, sample_indexer, C)
        return values_sum / values_count.unsqueeze(-1)
    elif op == 'sum':
        from system.ext import groupby_sum
        values_sum, _ = groupby_sum(sample_values, sample_indexer, C)
        return values_sum
    else:
        raise NotImplementedError


def fix_weight_norm_pickle(net: torch.nn.Module):
    from torch.nn.utils.weight_norm import WeightNorm
    for mdl in net.modules():
        fix_name = None
        if isinstance(mdl, torch.nn.Linear):
            for k, hook in mdl._forward_pre_hooks.items():
                if isinstance(hook, WeightNorm):
                    fix_name = hook.name
        if fix_name is not None:
            delattr(mdl, fix_name)


def overlay_sparse(locs_1: torch.Tensor, locs_2: torch.Tensor, spatial_size: list):
    if len(locs_1) == 0 or len(locs_2) == 0:
        return torch.zeros((0), device=locs_1.device), torch.zeros((0), device=locs_2.device)
    batch_size = torch.max(locs_1[:,-1]).item()+1
    assert batch_size == torch.max(locs_2[:,-1]).item()+1

    locs_1_linear = (locs_1[:,0] * spatial_size[1] * spatial_size[2] + locs_1[:,1] * spatial_size[2] + locs_1[:,2]) * batch_size + locs_1[:,3]
    locs_2_linear = (locs_2[:,0] * spatial_size[1] * spatial_size[2] + locs_2[:,1] * spatial_size[2] + locs_2[:,2]) * batch_size + locs_2[:,3]
    indicator_1 = torch.zeros(int(spatial_size[0]*spatial_size[1]*spatial_size[2]*batch_size), dtype=torch.long, device=locs_1.device)
    indicator_2 = indicator_1.clone()
    indicator_1[locs_1_linear] = torch.arange(locs_1_linear.shape[0], device=locs_1.device) + 1  # index valid locations
    indicator_2[locs_2_linear] = torch.arange(locs_2_linear.shape[0], device=locs_2.device) + 1  # index valid locations
    inds = torch.nonzero((indicator_1 > 0) & (indicator_2 > 0)).squeeze(1)  # pay attention to aligned valid locations
    ind_1 = indicator_1[inds] - 1
    ind_2 = indicator_2[inds] - 1
    return ind_1, ind_2


# def sparse_to_dense(locs: torch.Tensor, feats: torch.Tensor, spatial_size: list):
#     x = scn.InputLayer(3, spatial_size, mode=0)([locs.long(), feats.float()])
#     x = scn.SparseToDense(3, feats.shape[-1])(x)
#     return x


def generate_occ_dense(occ_locs: torch.Tensor, input_locs: torch.Tensor, spatial_size: list):
    assert occ_locs.shape[1] == input_locs.shape[1]

    # Input may not be in the batch mode
    if occ_locs.shape[1] == 3:
        batch_size = 1
        occ_locs = torch.cat([occ_locs, torch.zeros((occ_locs.shape[0], 1), device=occ_locs.device)], dim=1).long()
        input_locs = torch.cat([input_locs, torch.zeros((input_locs.shape[0], 1), device=input_locs.device)], dim=1).long()
    else:
        batch_size = torch.max(occ_locs[:,-1]).item()+1

    # Far-away locations are unknown
    max_bound = torch.amax(occ_locs[:,0:-1], dim=0) + 2
    max_bound = torch.min(max_bound, torch.tensor(spatial_size, device=occ_locs.device))
    min_bound = torch.amin(occ_locs[:,0:-1], dim=0) - 2
    min_bound = torch.max(min_bound, torch.tensor([0,0,0], device=occ_locs.device))
    known_locs = torch.tensor([[i,j,k] for i in range(min_bound[0].item(), max_bound[0].item()) \
                                       for j in range(min_bound[1].item(), max_bound[1].item()) \
                                       for k in range(min_bound[2].item(), max_bound[2].item()) ])

    # Locations near the GT occupied locations
    dist = [-1, 0, 1]
    known_neighbor_offsets = torch.tensor([[i,j,k] for i in dist for j in dist for k in dist], device=occ_locs.device)
    extend_locs = occ_locs.unsqueeze(0).repeat(len(known_neighbor_offsets), 1, 1)
    for i in range(len(known_neighbor_offsets)):
        extend_locs[i, :, 0:-1] += known_neighbor_offsets[i]
    extend_locs = torch.unique(extend_locs.reshape(-1, occ_locs.shape[-1]), dim=0)
    valid_mask = torch.logical_and(torch.all(extend_locs[:, 0:-1] <  torch.tensor(spatial_size, device=occ_locs.device), dim=1), \
                     torch.all(extend_locs[:, 0:-1] >=  torch.tensor([0, 0, 0], device=occ_locs.device), dim=1))  # in-bound mask
    extend_locs = extend_locs[valid_mask]

    # GT dense occupancy volume
    dense_occ = -1 * torch.ones(batch_size, 1, spatial_size[0], spatial_size[1], spatial_size[2], device=occ_locs.device)  # unknown
    dense_known_mask = sparse_to_dense(known_locs, torch.ones(known_locs.shape[0], 1, device=known_locs.device), spatial_size) > 0
    dense_occ_mask = sparse_to_dense(occ_locs, torch.ones(occ_locs.shape[0], 1, device=occ_locs.device), spatial_size) > 0
    dense_occ[dense_known_mask] = 0  # empty
    dense_occ[dense_occ_mask] = 1  # occupied

    # Dense weights
    dense_weights = 0.001 * torch.ones(batch_size, 1, spatial_size[0], spatial_size[1], spatial_size[2], device=occ_locs.device)
    dense_extend_mask = sparse_to_dense(extend_locs, torch.ones(extend_locs.shape[0], 1, device=occ_locs.device), spatial_size) > 0
    dense_input_mask = sparse_to_dense(input_locs, torch.ones(input_locs.shape[0], 1, device=input_locs.device), spatial_size) > 0
    dense_weights[dense_extend_mask] = 1  # locations near GT occupied locations, assign more weights
    dense_weights[dense_input_mask] = 1  # already occupied in input

    return dense_occ, dense_weights


def generate_multi_level_occ_dense(dense_occ: torch.Tensor, dense_weights: torch.Tensor, num_levels: int):
    # Upsampling
    dense_occ_list = [dense_occ]
    dense_weights_list = [dense_weights]
    with torch.no_grad():
        for i in range(num_levels - 1):
            up_sampled_dense_occ = torch.nn.MaxPool3d(kernel_size=2)(dense_occ_list[-1])
            up_sampled_dense_weights = torch.nn.MaxPool3d(kernel_size=2)(dense_weights_list[-1])
            dense_occ_list.append(up_sampled_dense_occ)
            dense_weights_list.append(up_sampled_dense_weights)

    return dense_occ_list, dense_weights_list

