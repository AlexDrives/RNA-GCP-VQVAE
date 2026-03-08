"""Node feature computation functions."""
from typing import List, Union

import torch
import torch.nn.functional as F
from graphein.protein.tensor.angles import alpha, dihedrals, kappa
from graphein.protein.tensor.data import Protein, ProteinBatch
from graphein.protein.tensor.types import AtomTensor, CoordTensor
try:  # Optional dependency for Hydra-style configs
    from omegaconf import ListConfig  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - fallback when OmegaConf is unavailable
    ListConfig = list  # type: ignore
from torch_geometric.data import Batch, Data

from ..typecheck import jaxtyped, typechecker
from ..types import OrientationTensor, ScalarNodeFeature

from .sequence_features import amino_acid_one_hot
from .utils import _normalize


@jaxtyped(typechecker=typechecker)
def compute_scalar_node_features(
    x: Union[Batch, Data, Protein, ProteinBatch],
    node_features: Union[ListConfig, List[ScalarNodeFeature]],
) -> torch.Tensor:
    """
    Factory function for node features.

    .. seealso::
        :py:class:`models.gcpnet.types.ScalarNodeFeature` for a list of node
        features that can be computed.

    This function operates on a :py:class:`torch_geometric.data.Data` or
    :py:class:`torch_geometric.data.Batch` object and computes the requested
    node features.

    :param x: :py:class:`~torch_geometric.data.Data` or
        :py:class:`~torch_geometric.data.Batch` protein object.
    :type x: Union[Data, Batch]
    :param node_features: List of node features to compute.
    :type node_features: Union[List[str], ListConfig]
    :return: Tensor of node features of shape (``N x F``), where ``N`` is the
        number of nodes and ``F`` is the number of features.
    :rtype: torch.Tensor
    """
    feats = []
    for feature in node_features:
        if feature == "amino_acid_one_hot":
            feats.append(amino_acid_one_hot(x, num_classes=23))
        elif feature == "rna_base_one_hot":
            feats.append(amino_acid_one_hot(x, num_classes=5))
        elif feature == "rna_purine_pyrimidine":
            # residue_type index convention for RNA is expected as:
            # A=0, U=1, C=2, G=3, X=4. Purines are A/G -> 1, others -> 0.
            purine_mask = ((x.residue_type == 0) | (x.residue_type == 3)).float()
            feats.append(purine_mask.unsqueeze(-1))
        elif feature == "rna_backbone_dihedrals_sincos":
            feats.append(rna_backbone_dihedrals_sincos(x))
        elif feature == "alpha":
            feats.append(alpha(x.coords, x.batch, rad=True, embed=True))
        elif feature == "kappa":
            feats.append(kappa(x.coords, x.batch, rad=True, embed=True))
        elif feature == "dihedrals":
            feats.append(dihedrals(x.coords, x.batch, rad=True, embed=True))
        elif feature == "sequence_positional_encoding":
            continue
        else:
            raise ValueError(f"Node feature {feature} not recognised.")
    feats = [feat.unsqueeze(1) if feat.ndim == 1 else feat for feat in feats]
    # Return concatenated features or original features if no features were computed
    return torch.cat(feats, dim=1) if feats else x.x


def _dihedral_sincos(
    p0: torch.Tensor,
    p1: torch.Tensor,
    p2: torch.Tensor,
    p3: torch.Tensor,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute sin/cos(dihedral) for point quadruplets with robust invalid masking."""
    b0 = p1 - p0
    b1 = p2 - p1
    b2 = p3 - p2

    b1_norm = torch.linalg.norm(b1, dim=-1)
    b1_unit = b1 / b1_norm.unsqueeze(-1).clamp_min(eps)

    v = b0 - torch.sum(b0 * b1_unit, dim=-1, keepdim=True) * b1_unit
    w = b2 - torch.sum(b2 * b1_unit, dim=-1, keepdim=True) * b1_unit

    v_norm = torch.linalg.norm(v, dim=-1)
    w_norm = torch.linalg.norm(w, dim=-1)

    x_term = torch.sum(v * w, dim=-1)
    y_term = torch.sum(torch.cross(b1_unit, v, dim=-1) * w, dim=-1)
    angle = torch.atan2(y_term, x_term)

    finite_mask = (
        torch.isfinite(p0).all(dim=-1)
        & torch.isfinite(p1).all(dim=-1)
        & torch.isfinite(p2).all(dim=-1)
        & torch.isfinite(p3).all(dim=-1)
    )
    valid = finite_mask & (b1_norm > eps) & (v_norm > eps) & (w_norm > eps)

    sin_out = torch.zeros_like(angle)
    cos_out = torch.zeros_like(angle)
    if valid.any():
        sin_out[valid] = torch.sin(angle[valid])
        cos_out[valid] = torch.cos(angle[valid])
    return sin_out, cos_out


def rna_backbone_dihedrals_sincos(
    x: Union[Batch, Data, Protein, ProteinBatch],
) -> torch.Tensor:
    """
    Compute RNA backbone torsions alpha..zeta as per-residue sin/cos features.

    Atom order in ``x.rna_backbone_coords`` is expected to be:
    [P, O5', C5', C4', C3', O3'] with shape (N, 6, 3).
    """
    backbone = getattr(x, "rna_backbone_coords", None)
    if backbone is None:
        num_nodes = int(getattr(x, "num_nodes", 0))
        device = getattr(getattr(x, "coords", None), "device", torch.device("cpu"))
        return torch.zeros((num_nodes, 12), dtype=torch.float32, device=device)

    if not torch.is_tensor(backbone):
        backbone = torch.as_tensor(backbone, dtype=torch.float32)
    else:
        backbone = backbone.to(dtype=torch.float32)

    if backbone.ndim != 3 or backbone.size(1) < 6 or backbone.size(2) != 3:
        num_nodes = int(backbone.size(0)) if backbone.ndim >= 1 else 0
        return torch.zeros((num_nodes, 12), dtype=torch.float32, device=backbone.device)

    bb = backbone[:, :6, :]
    num_nodes = int(bb.size(0))
    if num_nodes == 0:
        return torch.zeros((0, 12), dtype=torch.float32, device=bb.device)

    batch = getattr(x, "batch", None)
    if batch is None or int(batch.numel()) != num_nodes:
        batch = torch.zeros(num_nodes, dtype=torch.long, device=bb.device)
    else:
        batch = batch.to(device=bb.device)

    idx = torch.arange(num_nodes, device=bb.device)
    prev_idx = torch.clamp(idx - 1, min=0)
    next_idx = torch.clamp(idx + 1, max=num_nodes - 1)

    same_prev = torch.zeros(num_nodes, dtype=torch.bool, device=bb.device)
    same_next = torch.zeros(num_nodes, dtype=torch.bool, device=bb.device)
    if num_nodes > 1:
        same_prev[1:] = batch[1:] == batch[:-1]
        same_next[:-1] = batch[:-1] == batch[1:]

    alpha_s, alpha_c = _dihedral_sincos(bb[prev_idx, 5], bb[:, 0], bb[:, 1], bb[:, 2])
    beta_s, beta_c = _dihedral_sincos(bb[:, 0], bb[:, 1], bb[:, 2], bb[:, 3])
    gamma_s, gamma_c = _dihedral_sincos(bb[:, 1], bb[:, 2], bb[:, 3], bb[:, 4])
    delta_s, delta_c = _dihedral_sincos(bb[:, 2], bb[:, 3], bb[:, 4], bb[:, 5])
    epsilon_s, epsilon_c = _dihedral_sincos(bb[:, 3], bb[:, 4], bb[:, 5], bb[next_idx, 0])
    zeta_s, zeta_c = _dihedral_sincos(bb[:, 4], bb[:, 5], bb[next_idx, 0], bb[next_idx, 1])

    alpha_s = torch.where(same_prev, alpha_s, torch.zeros_like(alpha_s))
    alpha_c = torch.where(same_prev, alpha_c, torch.zeros_like(alpha_c))
    epsilon_s = torch.where(same_next, epsilon_s, torch.zeros_like(epsilon_s))
    epsilon_c = torch.where(same_next, epsilon_c, torch.zeros_like(epsilon_c))
    zeta_s = torch.where(same_next, zeta_s, torch.zeros_like(zeta_s))
    zeta_c = torch.where(same_next, zeta_c, torch.zeros_like(zeta_c))

    return torch.stack(
        [
            alpha_s,
            alpha_c,
            beta_s,
            beta_c,
            gamma_s,
            gamma_c,
            delta_s,
            delta_c,
            epsilon_s,
            epsilon_c,
            zeta_s,
            zeta_c,
        ],
        dim=-1,
    )


@jaxtyped(typechecker=typechecker)
def compute_vector_node_features(
    x: Union[Batch, Data, Protein, ProteinBatch],
    vector_features: Union[ListConfig, List[str]],
) -> Union[Batch, Data, Protein, ProteinBatch]:
    """Factory function for vector features.

    Currently implemented vector features are:

        - ``orientation``: Orientation of each node in the protein backbone
        - ``virtual_cb_vector``: Virtual CB vector for each node in the protein
        backbone


    """
    vector_node_features = []
    for feature in vector_features:
        if feature == "orientation":
            vector_node_features.append(orientations(x.coords, x._slice_dict["coords"]))
        elif feature == "rna_orientation":
            vector_node_features.append(rna_orientations(x.coords))
        elif feature == "virtual_cb_vector":
            raise NotImplementedError("Virtual CB vector not implemented yet.")
        else:
            raise ValueError(f"Vector feature {feature} not recognised.")
    x.x_vector_attr = torch.cat(vector_node_features, dim=0)
    return x


@jaxtyped(typechecker=typechecker)
def orientations(
    X: Union[CoordTensor, AtomTensor], coords_slice_index: torch.Tensor, ca_idx: int = 1
) -> OrientationTensor:
    if X.ndim == 3:
        X = X[:, ca_idx, :]

    # NOTE: the first item in the coordinates slice index is always 0,
    # and the last item is always the node count of the batch
    batch_num_nodes = X.shape[0]
    slice_index = coords_slice_index[1:] - 1
    last_node_index = slice_index[:-1]
    first_node_index = slice_index[:-1] + 1

    # NOTE: all of the last (first) nodes in a subgraph have their
    # forward (backward) vectors set to a padding value (i.e., 0.0)
    # to mimic feature construction behavior with single input graphs
    forward_slice = X[1:] - X[:-1]
    backward_slice = X[:-1] - X[1:]

    if forward_slice.numel() > 0 and last_node_index.numel() > 0:
        max_forward_idx = forward_slice.size(0) - 1
        # zero the forward vectors for last nodes in each subgraph without boolean masks (torch.compile friendly)
        valid_forward_idx = last_node_index.clamp_min(0).clamp_max(max_forward_idx).to(X.device)
        forward_slice.index_fill_(0, valid_forward_idx, 0.0)

    if backward_slice.numel() > 0 and first_node_index.numel() > 0:
        max_backward_idx = backward_slice.size(0) - 1
        # zero the backward vectors for first nodes in each subgraph
        valid_backward_idx = (first_node_index - 1).clamp_min(0).clamp_max(max_backward_idx).to(X.device)
        backward_slice.index_fill_(0, valid_backward_idx, 0.0)

    # NOTE: padding first and last nodes with zero vectors does not impact feature normalization
    forward = _normalize(forward_slice)
    backward = _normalize(backward_slice)
    forward = F.pad(forward, [0, 0, 0, 1])
    backward = F.pad(backward, [0, 0, 1, 0])
    orientations = torch.cat((forward.unsqueeze(-2), backward.unsqueeze(-2)), dim=-2)

    # optionally debug/verify the orientations
    # last_node_indices = torch.cat((last_node_index, torch.tensor([batch_num_nodes - 1])), dim=0)
    # first_node_indices = torch.cat((torch.tensor([0]), first_node_index), dim=0)
    # intermediate_node_indices_mask = torch.ones(batch_num_nodes, device=X.device, dtype=torch.bool)
    # intermediate_node_indices_mask[last_node_indices] = False
    # intermediate_node_indices_mask[first_node_indices] = False
    # assert not orientations[last_node_indices][:, 0].any() and orientations[last_node_indices][:, 1].any()
    # assert orientations[first_node_indices][:, 0].any() and not orientations[first_node_indices][:, 1].any()
    # assert orientations[intermediate_node_indices_mask][:, 0].any() and orientations[intermediate_node_indices_mask][:, 1].any()

    return orientations


@jaxtyped(typechecker=typechecker)
def rna_orientations(
    X: Union[CoordTensor, AtomTensor],
) -> OrientationTensor:
    """
    RNA residue orientation from rigid-body atoms [C4', C1', N1/N9].

    Returns a tensor of shape (n_nodes, 2, 3):
    - first vector: C1' -> N1/N9 (primary)
    - second vector: C1' -> C4' (secondary)
    """
    if X.ndim != 3 or X.size(1) < 3:
        raise ValueError("RNA orientation expects coordinate tensor of shape (N, 3, 3).")

    c4p = X[:, 0, :]
    c1p = X[:, 1, :]
    n_atom = X[:, 2, :]

    primary = _normalize(n_atom - c1p)
    secondary = _normalize(c4p - c1p)
    return torch.cat((primary.unsqueeze(-2), secondary.unsqueeze(-2)), dim=-2)
