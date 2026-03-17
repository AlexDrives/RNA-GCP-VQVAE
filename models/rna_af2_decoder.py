import torch
import torch.nn as nn
import torch.nn.functional as F

from models.gcpnet.geometry import Affine3D, RotationMatrix
from models.gcpnet.layers.structure_proj import RNA_BB_COORDINATES


def _rotation_6d_to_matrix(rot_6d: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    first = rot_6d[..., :3]
    second = rot_6d[..., 3:]

    basis_x = F.normalize(first, dim=-1, eps=eps)
    second = second - torch.sum(basis_x * second, dim=-1, keepdim=True) * basis_x
    basis_y = F.normalize(second, dim=-1, eps=eps)
    basis_z = torch.cross(basis_x, basis_y, dim=-1)

    return torch.stack([basis_x, basis_y, basis_z], dim=-1)


def _affine_from_origin_points(
    origin: torch.Tensor,
    x_point: torch.Tensor,
    xy_point: torch.Tensor,
    eps: float = 1e-8,
) -> Affine3D:
    x_axis = x_point - origin
    xy_plane = xy_point - origin
    rotation = RotationMatrix.from_graham_schmidt(x_axis, xy_plane, eps=eps)
    return Affine3D(trans=origin, rot=rotation)


def _affine_to_matrix(frames: Affine3D) -> torch.Tensor:
    rotation = frames.rot.as_matrix().tensor.unflatten(-1, (3, 3))
    return torch.cat([rotation, frames.trans.unsqueeze(-1)], dim=-1)


class InvariantPointAttentionNoPair(nn.Module):
    def __init__(
        self,
        num_channel: int,
        num_head: int,
        num_scalar_qk: int,
        num_scalar_v: int,
        num_point_qk: int,
        num_point_v: int,
    ) -> None:
        super().__init__()
        self.num_head = num_head
        self.num_scalar_qk = num_scalar_qk
        self.num_scalar_v = num_scalar_v
        self.num_point_qk = num_point_qk
        self.num_point_v = num_point_v
        self.scalar_scale = num_scalar_qk ** -0.5

        self.q_scalar = nn.Linear(num_channel, num_head * num_scalar_qk)
        self.k_scalar = nn.Linear(num_channel, num_head * num_scalar_qk)
        self.v_scalar = nn.Linear(num_channel, num_head * num_scalar_v)

        self.q_point = nn.Linear(num_channel, num_head * num_point_qk * 3)
        self.k_point = nn.Linear(num_channel, num_head * num_point_qk * 3)
        self.v_point = nn.Linear(num_channel, num_head * num_point_v * 3)

        self.point_weights = nn.Parameter(torch.zeros(num_head, num_point_qk))
        output_dim = num_head * (num_scalar_v + num_point_v * 4)
        self.output_projection = nn.Linear(output_dim, num_channel)

    def forward(
        self,
        single_repr: torch.Tensor,
        frames: Affine3D,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = single_repr.shape
        valid_mask = valid_mask.to(dtype=torch.bool, device=single_repr.device)

        q_scalar = self.q_scalar(single_repr).view(
            batch_size, seq_len, self.num_head, self.num_scalar_qk
        )
        k_scalar = self.k_scalar(single_repr).view(
            batch_size, seq_len, self.num_head, self.num_scalar_qk
        )
        v_scalar = self.v_scalar(single_repr).view(
            batch_size, seq_len, self.num_head, self.num_scalar_v
        )

        q_point_local = self.q_point(single_repr).view(
            batch_size, seq_len, self.num_head, self.num_point_qk, 3
        )
        k_point_local = self.k_point(single_repr).view(
            batch_size, seq_len, self.num_head, self.num_point_qk, 3
        )
        v_point_local = self.v_point(single_repr).view(
            batch_size, seq_len, self.num_head, self.num_point_v, 3
        )

        q_point_global = frames[..., None, None].apply(q_point_local)
        k_point_global = frames[..., None, None].apply(k_point_local)
        v_point_global = frames[..., None, None].apply(v_point_local)

        q_scalar = q_scalar.permute(0, 2, 1, 3)
        k_scalar = k_scalar.permute(0, 2, 1, 3)
        v_scalar = v_scalar.permute(0, 2, 1, 3)

        q_point_global = q_point_global.permute(0, 2, 1, 3, 4)
        k_point_global = k_point_global.permute(0, 2, 1, 3, 4)
        v_point_global = v_point_global.permute(0, 2, 1, 3, 4)

        scalar_logits = torch.einsum("bhid,bhjd->bhij", q_scalar, k_scalar) * self.scalar_scale

        point_delta = q_point_global[:, :, :, None, :, :] - k_point_global[:, :, None, :, :, :]
        point_dist2 = torch.sum(point_delta.square(), dim=-1)
        point_weights = F.softplus(self.point_weights).view(1, self.num_head, 1, 1, self.num_point_qk)
        point_logits = -0.5 * torch.sum(point_dist2 * point_weights, dim=-1)

        attention_logits = scalar_logits + point_logits
        key_mask = valid_mask[:, None, None, :]
        attention_logits = attention_logits.masked_fill(~key_mask, -1e4)
        attention = torch.softmax(attention_logits, dim=-1)
        attention = attention * valid_mask[:, None, :, None].to(attention.dtype)

        scalar_out = torch.einsum("bhij,bhjd->bhid", attention, v_scalar)
        point_out_global = torch.einsum("bhij,bhjpd->bhipd", attention, v_point_global)
        point_out_global = point_out_global.permute(0, 2, 1, 3, 4)
        point_out_local = frames[..., None, None].invert().apply(point_out_global)
        point_norm = torch.linalg.norm(point_out_local, dim=-1)

        scalar_out = scalar_out.permute(0, 2, 1, 3).reshape(batch_size, seq_len, -1)
        point_out_local = point_out_local.reshape(batch_size, seq_len, -1)
        point_norm = point_norm.reshape(batch_size, seq_len, -1)

        combined = torch.cat([scalar_out, point_out_local, point_norm], dim=-1)
        update = self.output_projection(combined)
        return update * valid_mask.unsqueeze(-1).to(update.dtype)


class StructureTransition(nn.Module):
    def __init__(self, num_channel: int, hidden_factor: int) -> None:
        super().__init__()
        hidden_dim = num_channel * hidden_factor
        self.net = nn.Sequential(
            nn.Linear(num_channel, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_channel),
        )

    def forward(self, single_repr: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        update = self.net(single_repr)
        return update * valid_mask.unsqueeze(-1).to(update.dtype)


class RelativeRigidUpdate(nn.Module):
    def __init__(self, num_channel: int, trans_scale_factor: float) -> None:
        super().__init__()
        self.trans_scale_factor = trans_scale_factor
        self.proj = nn.Linear(num_channel, 9)

    def forward(
        self,
        single_repr: torch.Tensor,
        frames: Affine3D,
        valid_mask: torch.Tensor,
    ) -> Affine3D:
        update = self.proj(single_repr)
        delta_t = update[..., :3] * self.trans_scale_factor
        delta_rot = _rotation_6d_to_matrix(update[..., 3:])
        delta_frame = Affine3D(
            trans=delta_t,
            rot=RotationMatrix(delta_rot),
        )
        delta_frame = delta_frame.mask(~valid_mask.to(dtype=torch.bool, device=single_repr.device))
        next_frames = frames.compose(delta_frame)
        return next_frames.mask(~valid_mask.to(dtype=torch.bool, device=single_repr.device))


class RNAStructureBlock(nn.Module):
    def __init__(self, decoder_configs) -> None:
        super().__init__()
        num_channel = int(decoder_configs.num_channel)
        self.ipa = InvariantPointAttentionNoPair(
            num_channel=num_channel,
            num_head=int(decoder_configs.num_head),
            num_scalar_qk=int(decoder_configs.num_scalar_qk),
            num_scalar_v=int(decoder_configs.num_scalar_v),
            num_point_qk=int(decoder_configs.num_point_qk),
            num_point_v=int(decoder_configs.num_point_v),
        )
        self.ipa_dropout = nn.Dropout(float(decoder_configs.dropout))
        self.ipa_norm = nn.LayerNorm(num_channel)

        self.transition = StructureTransition(
            num_channel=num_channel,
            hidden_factor=int(decoder_configs.transition_hidden_factor),
        )
        self.transition_dropout = nn.Dropout(float(decoder_configs.dropout))
        self.transition_norm = nn.LayerNorm(num_channel)

        self.frame_update = RelativeRigidUpdate(
            num_channel=num_channel,
            trans_scale_factor=float(decoder_configs.trans_scale_factor),
        )

    def forward(
        self,
        single_repr: torch.Tensor,
        frames: Affine3D,
        valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, Affine3D]:
        ipa_update = self.ipa(single_repr, frames, valid_mask)
        single_repr = self.ipa_norm(single_repr + self.ipa_dropout(ipa_update))

        transition_update = self.transition(single_repr, valid_mask)
        single_repr = self.transition_norm(
            single_repr + self.transition_dropout(transition_update)
        )
        single_repr = single_repr * valid_mask.unsqueeze(-1).to(single_repr.dtype)

        frames = self.frame_update(single_repr, frames, valid_mask)
        return single_repr, frames


class RNAAF2Decoder(nn.Module):
    def __init__(self, configs, decoder_configs) -> None:
        super().__init__()
        self.max_length = int(configs.model.max_length)
        self.decoder_causal = False
        self.input_dim = int(getattr(decoder_configs, "input_dim", configs.model.vqvae.vector_quantization.dim))
        self.num_channel = int(decoder_configs.num_channel)
        self.num_layer = int(decoder_configs.num_layer)
        self.share_weights = bool(getattr(decoder_configs, "share_weights", True))

        configured_rna_template = getattr(decoder_configs, "rna_rigid_template_coords", None)
        if configured_rna_template is not None:
            template_coords = torch.as_tensor(configured_rna_template, dtype=torch.float32)
            if template_coords.shape != (3, 3):
                raise ValueError("decoder_configs.rna_rigid_template_coords must have shape (3, 3).")
        else:
            template_coords = RNA_BB_COORDINATES
        self.register_buffer("template_coords", template_coords.to(torch.float32))

        self.input_proj = nn.Linear(self.input_dim, self.num_channel, bias=False)
        self.input_norm = nn.LayerNorm(self.num_channel)

        if self.share_weights:
            self.structure_block = RNAStructureBlock(decoder_configs)
            self.structure_blocks = None
        else:
            self.structure_block = None
            self.structure_blocks = nn.ModuleList(
                [RNAStructureBlock(decoder_configs) for _ in range(self.num_layer)]
            )

    def _iter_blocks(self):
        if self.share_weights:
            return [self.structure_block for _ in range(self.num_layer)]
        return list(self.structure_blocks)

    def _frame_init(self, shape: tuple[int, ...], device: torch.device, dtype: torch.dtype) -> Affine3D:
        return Affine3D.identity(
            shape,
            rotation_type=RotationMatrix,
            device=device,
            dtype=dtype,
        )

    def _coords_from_frames(self, frames: Affine3D, valid_mask: torch.Tensor) -> torch.Tensor:
        coords_local = self.template_coords.to(device=frames.device, dtype=frames.dtype).reshape(1, 1, 3, 3)
        pred_xyz = frames[..., None].apply(coords_local)
        return pred_xyz * valid_mask.unsqueeze(-1).unsqueeze(-1).to(pred_xyz.dtype)

    def _stop_rotation_gradient(self, frames: Affine3D) -> Affine3D:
        # Match AF2's affine.apply_rotation_tensor_fn(stop_gradient): keep using
        # the updated frame geometry, but prevent later blocks from backpropagating
        # through the current rotation tensor.
        return Affine3D(trans=frames.trans, rot=frames.rot.detach())

    def forward(
        self,
        structure_tokens: torch.Tensor,
        mask: torch.Tensor,
        *,
        true_lengths=None,
    ):
        del true_lengths
        valid_mask = mask.to(dtype=torch.bool, device=structure_tokens.device)
        batch_size, seq_len, _ = structure_tokens.shape

        single_repr = self.input_proj(structure_tokens)
        single_repr = self.input_norm(single_repr)
        single_repr = single_repr * valid_mask.unsqueeze(-1).to(single_repr.dtype)

        frames = self._frame_init((batch_size, seq_len), device=structure_tokens.device, dtype=structure_tokens.dtype)
        frames = frames.mask(~valid_mask)

        frame_traj = []
        coord_traj = []

        # Match AF2 structure module semantics: run the fold block num_layer times,
        # optionally reusing the same block weights, without an extra recycle loop.
        for block in self._iter_blocks():
            single_repr, frames = block(single_repr, frames, valid_mask)
            coords = self._coords_from_frames(frames, valid_mask)
            frame_traj.append(_affine_to_matrix(frames))
            coord_traj.append(coords)
            frames = self._stop_rotation_gradient(frames)

        final_frames = frame_traj[-1]
        outputs = coord_traj[-1]

        return {
            "outputs": outputs,
            "final_frames": final_frames,
            "frame_traj": torch.stack(frame_traj, dim=1),
            "coord_traj": torch.stack(coord_traj, dim=1),
            "num_recycles": 0,
            "dir_loss_logits": None,
            "dist_loss_logits": None,
        }


__all__ = [
    "RNAAF2Decoder",
    "_affine_from_origin_points",
    "_affine_to_matrix",
]




