from typing import Literal

from torch_geometric.data import Batch

from models.gcpnet.typecheck import jaxtyped, typechecker


@jaxtyped(typechecker=typechecker)
def transform_representation(
    batch: Batch, representation_type: Literal["CA", "C1P"]
) -> Batch:
    """Assign ``batch.pos`` to the configured anchor coordinates.

    Supported anchors:
    - ``CA``: protein C-alpha
    - ``C1P``: RNA C1' anchor
    """

    if representation_type not in ("CA", "C1P"):  # pragma: no cover - defensive guard
        raise ValueError(
            "Trimmed ProteinWorkshop only supports 'CA' and 'C1P' representations"
        )

    # Both supported representations use atom index 1 in their local templates.
    batch.pos = batch.coords[:, 1, :]
    return batch


__all__ = ["transform_representation"]
