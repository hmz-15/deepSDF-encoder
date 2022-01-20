import torch
import torch.nn.functional as F
import torch.distributions


# ----------------------------------
# DATA CRITERIA:
# ----------------------------------

def l1_loss(args, info: dict, pd_sdf: torch.Tensor, gt_sdf: torch.Tensor, **kwargs):
    """
    L1 Loss: make pd_sdf and gt_sdf as close as possible.
    Theoretically, this loss is way easier than siren loss.
    But siren's advantage is that:
        1) it does not perturbed samples.
        2) it does not need perturbed gt_sdf.
    :return:
    """
    if args.enforce_minmax:
        gt_sdf = torch.clamp(gt_sdf, -args.clamping_distance, args.clamping_distance)
        pd_sdf = torch.clamp(pd_sdf, -args.clamping_distance, args.clamping_distance)

    sdf_loss = (gt_sdf.squeeze() - pd_sdf).abs().sum() / info["num_sdf_samples"]
    return {
        'sdf': sdf_loss
    }


def neg_log_likelihood(args, info: dict, pd_sdf: torch.Tensor, pd_sdf_std: torch.Tensor,
                       gt_sdf: torch.Tensor, loss_name: str = 'll', **kwargs):
    """
    Negative log likelihood of gt data under predicted gaussian distribution.
    """
    if len(pd_sdf) == 0:
        print("empty pd_sdf!")
    elif torch.any(torch.isnan(pd_sdf)).item():
        print(pd_sdf)
        print("pd is nan!")
    if args.enforce_minmax:
        gt_sdf = torch.clamp(gt_sdf, -args.clamping_distance, args.clamping_distance)
        pd_sdf = torch.clamp(pd_sdf, -args.clamping_distance, args.clamping_distance)

    pd_dist = torch.distributions.Normal(loc=pd_sdf.squeeze(), scale=pd_sdf_std.squeeze())
    # print(pd_dist.log_prob(gt_sdf.squeeze()))
    sdf_loss = -pd_dist.log_prob(gt_sdf.squeeze()).sum() / info["num_sdf_samples"]
    return {
        loss_name: sdf_loss
    }


def siren_loss(args, info: dict, pd_sdf: torch.Tensor, coords: torch.Tensor, gt_sdf: torch.Tensor, **kwargs):
    """
    SIREN Loss: Boundary sdf value = 0 +
                Non-boundary sdf value != 0 +
                boundary normal = gt-normal +
                gradient's norm is 1
    :param pd_sdf: (B, 1)
    :param coords: (B, 3): The computation graph from this to pd_sdf should be provided to compute the gradient!
    :param gt_sdf: (B, 3) nx, ny, nz
    :return: loss-dict differentiable w.r.t. pd_sdf.
    """
    # Note: Here torch.ones_like(pd_sdf) is okay because each sample in the batch is not relevant.
    # So that the derivative of pd_sdf in batch2 (S2) w.r.t. the coords in batch1 is always 0!
    # Hence d(S1+...+SB)/d(xyz) = d(S1)/d(xyz)
    # To compute a real full Jacobian, each element in coords should be evaluated separately!
    pd_sdf_grad = torch.autograd.grad(pd_sdf, [coords],
                                      grad_outputs=torch.ones_like(pd_sdf), create_graph=True)[0]

    gt_normals = gt_sdf
    gt_sdf = torch.sum(gt_sdf.abs(), dim=-1, keepdim=True) > 1e-6

    zero_loss_branch = torch.zeros_like(pd_sdf, requires_grad=False)

    sdf_in_loss = torch.where(gt_sdf, pd_sdf, zero_loss_branch)
    sdf_out_loss = torch.where(gt_sdf, zero_loss_branch, torch.exp(-1e2 * torch.abs(pd_sdf)))
    normal_loss = torch.where(gt_sdf, 1. - F.cosine_similarity(pd_sdf_grad, gt_normals, dim=-1).unsqueeze(1),
                              zero_loss_branch)
    eikonal_loss = torch.abs(pd_sdf_grad.norm(dim=-1) - 1)

    return {
        'sdf_in': torch.abs(sdf_in_loss).sum() / info["num_sdf_samples"] * args.siren_sdf_in,
        'sdf_out': sdf_out_loss.sum() / info["num_sdf_samples"] * args.siren_sdf_out,
        'normal': normal_loss.sum() / info["num_sdf_samples"] * args.siren_normal,
        'eikonal': eikonal_loss.sum() / info["num_sdf_samples"] * args.siren_eikonal
    }


# ----------------------------------
# REG CRITERIA:
# ----------------------------------

def reg_loss(args, info: dict, latent_vecs: torch.Tensor, **kwargs):
    l2_size_loss = torch.sum(torch.norm(latent_vecs, dim=1))
    reg_loss = min(1, info["epoch"] * 3/ info["num_epochs"]) * l2_size_loss / latent_vecs.shape[0]
    return {
        'reg': reg_loss * args.code_reg_lambda
    }


def l2_loss(args, info: dict, latent_vecs: torch.Tensor, latent_vecs_ref: torch.Tensor, **kwargs):
    diff = latent_vecs - latent_vecs_ref
    # l2_loss = min(1, info["epoch"] * 3/ info["num_epochs"]) * torch.sum(torch.norm(diff, dim=1)) / latent_vecs.shape[0]
    l2_loss = torch.sum(torch.norm(diff, dim=1)) / latent_vecs.shape[0]
    return {
        'l2': l2_loss * args.code_diff_lambda
    }


def norm_l2_loss(args, info: dict, latent_vecs: torch.Tensor, latent_vecs_ref: torch.Tensor, **kwargs):
    diff = F.normalize(latent_vecs, dim=1) - F.normalize(latent_vecs_ref, dim=1)
    # l2_loss = min(1, info["epoch"] * 3/ info["num_epochs"]) * torch.sum(torch.norm(diff, dim=1)) / latent_vecs.shape[0]
    l2_loss = torch.sum(torch.norm(diff, dim=1)) / latent_vecs.shape[0]
    return {
        'l2': l2_loss * args.code_diff_lambda
    }


# ----------------------------------
# COMPLETION CRITERIA:
# ----------------------------------

def bce_loss(pd_occ: torch.Tensor, gt_occ: torch.Tensor, weights: torch.Tensor, **kwargs):
    mask = torch.logical_and(pd_occ > 0.0, gt_occ > -1)  # filter invalid predictions and unknown locs in ground truth
    loss = F.binary_cross_entropy_with_logits(pd_occ[mask].view(-1), gt_occ[mask].view(-1), weight=weights[mask].view(-1))
    return loss


def multi_level_bce_loss(args, pd_occ_list: list, gt_occ_list: list, weights_list: list, **kwargs):
    assert len(pd_occ_list) == len(gt_occ_list) == len(args.occ_weights)
    loss = 0
    for i in range(len(pd_occ_list)):
        loss = loss + bce_loss(pd_occ_list[i], gt_occ_list[i], weights_list[i]) * args.occ_weights[i]
    return {
        'bce': loss
    }


