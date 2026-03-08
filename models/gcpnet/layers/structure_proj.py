import torch
import torch.nn as nn

from ..geometry import Affine3D, RotationMatrix

# Protein local template [N, CA, C] with CA at origin.
PROTEIN_BB_COORDINATES = torch.tensor(
    [
        [0.5256, 1.3612, 0.0000],
        [0.0000, 0.0000, 0.0000],
        [-1.5251, 0.0000, 0.0000],
    ],
    dtype=torch.float32,
)

# RNA local rigid-body template [C4', C1', N1/N9] with C1' at origin.
# Values are fixed canonical coordinates used under the rigid-body approximation.
RNA_BB_COORDINATES = torch.tensor(
    [
        [-1.288004, 1.960215, 0.000000],
        [0.0000, 0.0000, 0.0000],
        [1.469997, 0.000000, 0.000000],
    ],
    dtype=torch.float32,
)


class Dim6RotStructureHead(nn.Module):
    """Predict backbone frames and coordinates from latent embeddings."""

    def __init__(
        self,
        input_dim: int,
        trans_scale_factor: float = 10.0,
        predict_torsion_angles: bool = False,
        template_coords: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.ffn1 = nn.Linear(input_dim, input_dim)
        self.activation_fn = nn.GELU()
        self.norm = nn.LayerNorm(input_dim)
        self.predict_torsion_angles = predict_torsion_angles
        projection_dim = 9 + (14 if predict_torsion_angles else 0)
        self.proj = nn.Linear(input_dim, projection_dim)
        self.trans_scale_factor = trans_scale_factor
        if template_coords is None:
            template_coords = PROTEIN_BB_COORDINATES
        if template_coords.shape != (3, 3):
            raise ValueError("template_coords must have shape (3, 3).")
        self.register_buffer("template_coords", template_coords.to(torch.float32))

    def forward(self, x: torch.Tensor, affine: Affine3D | None, affine_mask: torch.Tensor, **_kwargs):
        if affine is None:
            rigids = Affine3D.identity(
                x.shape[:-1],
                dtype=x.dtype,
                device=x.device,
                rotation_type=RotationMatrix,
            )
        else:
            rigids = affine

        x = self.ffn1(x)
        x = self.activation_fn(x)
        x = self.norm(x)

        if self.predict_torsion_angles:
            trans, vec_x, vec_y, _ = self.proj(x).split([3, 3, 3, 14], dim=-1)
        else:
            trans, vec_x, vec_y = self.proj(x).split([3, 3, 3], dim=-1)

        trans = trans * self.trans_scale_factor
        vec_x = vec_x / (vec_x.norm(dim=-1, keepdim=True) + 1e-5)
        vec_y = vec_y / (vec_y.norm(dim=-1, keepdim=True) + 1e-5)

        update = Affine3D.from_graham_schmidt(vec_x + trans, trans, vec_y + trans)
        rigids = rigids.compose(update.mask(affine_mask))

        coords_local = self.template_coords.to(x.device).reshape(1, 1, 3, 3)
        pred_xyz = rigids[..., None].apply(coords_local)

        return rigids.tensor, pred_xyz


__all__ = [
    "Dim6RotStructureHead",
    "PROTEIN_BB_COORDINATES",
    "RNA_BB_COORDINATES",
]
