from aseMolec.anaAtoms import find_molecs, wrap_molecs, split_molecs, scan_vol
import ase.io
import numpy as np
import torch

from typing import Optional

import torch


def _broadcast(src: torch.Tensor, other: torch.Tensor, dim: int):
    if dim < 0:
        dim = other.dim() + dim
    if src.dim() == 1:
        for _ in range(0, dim):
            src = src.unsqueeze(0)
    for _ in range(src.dim(), other.dim()):
        src = src.unsqueeze(-1)
    src = src.expand_as(other)
    return src


def scatter_sum(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = -1,
    out: Optional[torch.Tensor] = None,
    dim_size: Optional[int] = None,
    reduce: str = "sum",
) -> torch.Tensor:
    assert reduce == "sum"  # for now, TODO
    index = _broadcast(index, src, dim)
    if out is None:
        size = list(src.size())
        if dim_size is not None:
            size[dim] = dim_size
        elif index.numel() == 0:
            size[dim] = 0
        else:
            size[dim] = int(index.max()) + 1
        out = torch.zeros(size, dtype=src.dtype, device=src.device)
        return out.scatter_add_(dim, index, src)
    else:
        return out.scatter_add_(dim, index, src)


def scatter_std(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = -1,
    out: Optional[torch.Tensor] = None,
    dim_size: Optional[int] = None,
    unbiased: bool = True,
) -> torch.Tensor:
    if out is not None:
        dim_size = out.size(dim)

    if dim < 0:
        dim = src.dim() + dim

    count_dim = dim
    if index.dim() <= dim:
        count_dim = index.dim() - 1

    ones = torch.ones(index.size(), dtype=src.dtype, device=src.device)
    count = scatter_sum(ones, index, count_dim, dim_size=dim_size)

    index = _broadcast(index, src, dim)
    tmp = scatter_sum(src, index, dim, dim_size=dim_size)
    count = _broadcast(count, tmp, dim).clamp(1)
    mean = tmp.div(count)

    var = src - mean.gather(dim, index)
    var = var * var
    out = scatter_sum(var, index, dim, out, dim_size)

    if unbiased:
        count = count.sub(1).clamp_(1)
    out = out.div(count + 1e-6).sqrt()

    return out


def scatter_mean(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = -1,
    out: Optional[torch.Tensor] = None,
    dim_size: Optional[int] = None,
) -> torch.Tensor:
    out = scatter_sum(src, index, dim, out, dim_size)
    dim_size = out.size(dim)

    index_dim = dim
    if index_dim < 0:
        index_dim = index_dim + src.dim()
    if index.dim() <= index_dim:
        index_dim = index.dim() - 1

    ones = torch.ones(index.size(), dtype=src.dtype, device=src.device)
    count = scatter_sum(ones, index, index_dim, None, dim_size)
    count[count < 1] = 1
    count = _broadcast(count, out, dim)
    if out.is_floating_point():
        out.true_divide_(count)
    else:
        out.div_(count, rounding_mode="floor")
    return out


def add_cluster_stresses(ats_list, force_key):
    delta = 0.01
    for att in ats_list:
        mols = scan_vol(att, np.array([1.,1. + delta]))
        delta_pos = mols[1].get_positions() - mols[0].get_positions()
        work_done = np.sum(att.arrays[force_key] * delta_pos)
        att.info[f'cluster_stress_{force_key}'] = work_done / delta


def add_cluster_stess_analytic(ats_list, force_key):
    find_molecs(ats_list)
    for att in ats_list:
        forces = att.arrays[force_key]
        mol_ids = att.arrays['molID']
        forces_tensor = torch.tensor(forces)
        mol_ids_tensor = torch.tensor(mol_ids, dtype=torch.long)
        positions_tensor = torch.tensor(att.get_positions())
        molecule_centers = scatter_mean(
            positions_tensor, mol_ids_tensor, dim=0
        )
        molecule_forces = scatter_sum(
            forces_tensor, mol_ids_tensor, dim=0
        )
        energy = torch.sum(molecule_forces * molecule_centers)
        att.info[f'analytic_cluster_stress_{force_key}'] = energy.numpy()