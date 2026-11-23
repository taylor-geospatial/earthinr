# Copyright (c) Taylor Geospatial. All rights reserved.
# Licensed under the MIT License.

"""Tessellating the Earth (TTE) spherical Voronoi location encoder."""

import math
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as fn

from .._types import Path
from ._base import LocationEncoder, Weights, WeightsEnum
from ._utils import load_safetensors

_EARTHINR_ZOO_REPO = 'taylor-geospatial/earthinr-zoo'

_UPSTREAM_IMAGE_KEYS = frozenset(
    {
        'location.semantic_registers.img_to_logits.weight',
        'location.semantic_registers.img_to_logits.bias',
        'location.semantic_registers._img_temperature',
    }
)


class _ResLayer(nn.Module):
    """Residual fully connected block used in the TTE MLP head."""

    def __init__(self, hidden_dim: int, dropout: float = 0.5) -> None:
        """Initialize a new _ResLayer instance.

        Args:
            hidden_dim: Hidden feature dimensionality.
            dropout: Dropout probability applied between linear projections.
        """
        super().__init__()
        self.w1 = nn.Linear(hidden_dim, hidden_dim)
        self.w2 = nn.Linear(hidden_dim, hidden_dim)
        self.nonlin1 = nn.ReLU(inplace=True)
        self.nonlin2 = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        """Apply the residual block.

        Args:
            x: Input tensor of shape ``(B, hidden_dim)``.

        Returns:
            Output tensor of shape ``(B, hidden_dim)``.
        """
        y = self.w1(x)
        y = self.nonlin1(y)
        y = self.dropout(y)
        y = self.w2(y)
        y = self.nonlin2(y)
        return x + y


class SemanticRegisterAttention(nn.Module):
    """Location-side attention over shared global semantic register tokens."""

    def __init__(
        self,
        site_embed_dim: int = 384,
        output_dim: int = 512,
        n_registers: int = 64,
        temperature: float = 0.5,
        gate: float = 0.5,
        use_cosine_attention: bool = False,
    ) -> None:
        """Initialize a new SemanticRegisterAttention instance.

        Args:
            site_embed_dim: Dimensionality of aggregated Voronoi site embeddings.
            output_dim: Output dimensionality after register blending.
            n_registers: Number of learnable semantic register tokens.
            temperature: Initial attention temperature.
            gate: Fixed blend weight between skip site projection and registers.
            use_cosine_attention: Whether to use cosine similarity for register logits.
        """
        super().__init__()
        self.n_registers = n_registers
        self.site_embed_dim = site_embed_dim
        self.output_dim = output_dim
        self.gate = gate
        self.use_cosine_attention = use_cosine_attention

        registers = torch.empty(n_registers, output_dim)
        nn.init.orthogonal_(registers)
        self.registers = nn.Parameter(fn.normalize(registers, dim=-1))

        use_bias = not use_cosine_attention
        self.loc_to_logits = nn.Linear(site_embed_dim, n_registers, bias=use_bias)
        self.site_to_out: nn.Linear | None = (
            nn.Linear(site_embed_dim, output_dim)
            if site_embed_dim != output_dim
            else None
        )
        self.log_temperature = nn.Parameter(torch.tensor(math.log(temperature)))
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, site_embeddings: Tensor) -> Tensor:
        """Enhance site embeddings with semantic register attention.

        Args:
            site_embeddings: Tensor of shape ``(B, site_embed_dim)``.

        Returns:
            Layer-normalized blended tensor of shape ``(B, output_dim)``.
        """
        temp = torch.exp(self.log_temperature).clamp(min=0.01, max=10.0)
        if self.use_cosine_attention:
            site_norm = fn.normalize(site_embeddings, dim=-1)
            weight_norm = fn.normalize(self.loc_to_logits.weight, dim=1)
            logits = (site_norm @ weight_norm.T) / temp
        else:
            logits = self.loc_to_logits(site_embeddings) / temp

        attention = fn.softmax(logits, dim=-1)
        loc_attended = attention @ self.registers
        site_out = (
            site_embeddings
            if self.site_to_out is None
            else self.site_to_out(site_embeddings)
        )
        return self.norm((1.0 - self.gate) * site_out + self.gate * loc_attended)


class VoronoiLocationEncoder(nn.Module):
    """Spherical soft-Voronoi location encoder with semantic registers and MLP."""

    def __init__(
        self,
        n_sites: int = 4096,
        site_embed_dim: int = 384,
        tau_init: float = 45.0,
        output_dim: int = 512,
        hidden_dim: int = 512,
        n_reslayers: int = 2,
        dropout: float = 0.5,
        n_registers: int = 64,
        register_temperature: float = 0.5,
        register_fixed_gate: float = 0.5,
        use_cosine_attention: bool = False,
    ) -> None:
        """Initialize a new VoronoiLocationEncoder instance.

        Args:
            n_sites: Number of learnable Voronoi sites on the unit sphere.
            site_embed_dim: Dimensionality of each site embedding.
            tau_init: Initial inverse temperature for soft Voronoi assignment.
            output_dim: Output embedding dimensionality.
            hidden_dim: Hidden width of the residual MLP head.
            n_reslayers: Number of residual blocks in the MLP head.
            dropout: Dropout probability inside residual blocks.
            n_registers: Number of global semantic register tokens.
            register_temperature: Initial attention temperature for registers.
            register_fixed_gate: Blend weight for semantic register output.
            use_cosine_attention: Whether to use cosine register attention.
        """
        super().__init__()
        self.n_sites = n_sites
        self.site_embed_dim = site_embed_dim
        self.output_dim = output_dim
        self.n_registers = n_registers

        init_sites = self._fibonacci_lattice(n_sites)
        self._directions = nn.Parameter(fn.normalize(init_sites, dim=-1))
        self._log_tau = nn.Parameter(torch.full((n_sites,), math.log(tau_init)))
        self._embeddings = nn.Parameter(torch.randn(n_sites, site_embed_dim) * 0.02)

        self.semantic_registers = SemanticRegisterAttention(
            site_embed_dim=site_embed_dim,
            output_dim=output_dim,
            n_registers=n_registers,
            temperature=register_temperature,
            gate=register_fixed_gate,
            use_cosine_attention=use_cosine_attention,
        )

        layers: list[nn.Module] = [nn.Linear(output_dim, hidden_dim), nn.ReLU(True)]
        for _ in range(n_reslayers):
            layers.append(_ResLayer(hidden_dim, dropout=dropout))
        layers.append(nn.Linear(hidden_dim, output_dim, bias=False))
        self.mlp = nn.Sequential(*layers)

    @staticmethod
    def _fibonacci_lattice(n: int) -> Tensor:
        """Generate unit-sphere directions using a golden-ratio Fibonacci lattice.

        Args:
            n: Number of lattice points.

        Returns:
            Tensor of shape ``(n, 3)`` on the unit sphere.
        """
        indices = torch.arange(n, dtype=torch.float32)
        phi = (1.0 + math.sqrt(5.0)) / 2.0
        theta = 2.0 * math.pi * indices / phi
        z = 1.0 - (2.0 * indices + 1.0) / n
        r = torch.sqrt(torch.clamp(1.0 - z * z, min=0.0))
        x = r * torch.cos(theta)
        y = r * torch.sin(theta)
        return torch.stack((x, y, z), dim=-1)

    def forward(self, coords: Tensor) -> Tensor:
        """Encode ``(lat, lon)`` coordinates in degrees.

        Args:
            coords: Coordinate tensor of shape ``(B, 2)`` ordered as ``(lat, lon)``
                in degrees.

        Returns:
            Unnormalized location embeddings of shape ``(B, output_dim)``.
        """
        lat = torch.deg2rad(coords[:, 0])
        lon = torch.deg2rad(coords[:, 1])
        cos_lat = torch.cos(lat)
        sphere = torch.stack(
            (cos_lat * torch.cos(lon), cos_lat * torch.sin(lon), torch.sin(lat)), dim=-1
        )

        sites = fn.normalize(self._directions, dim=-1)
        scaled_sites = sites * torch.exp(self._log_tau).unsqueeze(-1)
        logits = sphere @ scaled_sites.T
        logits = logits - logits.max(dim=-1, keepdim=True).values
        weights = fn.softmax(logits, dim=-1)
        site_emb = weights @ self._embeddings

        attended = self.semantic_registers(site_emb)
        return self.mlp(attended)


class TTENet(nn.Module):
    """Top-level TTE module matching the pretrained checkpoint key hierarchy."""

    def __init__(
        self,
        n_sites: int = 4096,
        site_embed_dim: int = 384,
        output_dim: int = 512,
        hidden_dim: int = 512,
        n_reslayers: int = 2,
        n_registers: int = 64,
        register_fixed_gate: float = 0.5,
        use_cosine_attention: bool = False,
        normalize: bool = True,
    ) -> None:
        """Initialize a new TTENet instance.

        Args:
            n_sites: Number of learnable Voronoi sites.
            site_embed_dim: Per-site embedding dimension.
            output_dim: Output embedding dimension.
            hidden_dim: Residual MLP hidden width.
            n_reslayers: Number of residual layers in the MLP head.
            n_registers: Number of semantic register tokens.
            register_fixed_gate: Fixed skip-connection blend weight.
            use_cosine_attention: Whether to use cosine register attention.
            normalize: Whether to L2-normalize output embeddings.
        """
        super().__init__()
        self.normalize = normalize
        self.location = VoronoiLocationEncoder(
            n_sites=n_sites,
            site_embed_dim=site_embed_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            n_reslayers=n_reslayers,
            n_registers=n_registers,
            register_fixed_gate=register_fixed_gate,
            use_cosine_attention=use_cosine_attention,
        )

    def forward(self, coords: Tensor) -> Tensor:
        """Encode ``(lat, lon)`` coordinates in degrees.

        Args:
            coords: Coordinate tensor of shape ``(B, 2)`` in ``(lat, lon)`` degrees.

        Returns:
            Embedding tensor of shape ``(B, output_dim)``.
        """
        emb = self.location(coords)
        if self.normalize:
            return fn.normalize(emb, dim=-1)
        return emb


class TTE_Weights(WeightsEnum):  # noqa: N801
    """Pretrained weights for Tessellating the Earth (TTE)."""

    DEFAULT = Weights(
        repo_id=_EARTHINR_ZOO_REPO,
        filename='tte/model.safetensors',
        meta={
            'embed_dim': 512,
            'n_sites': 4096,
            'site_embed_dim': 384,
            'output_dim': 512,
            'hidden_dim': 512,
            'n_reslayers': 2,
            'n_registers': 64,
            'register_fixed_gate': 0.5,
            'use_cosine_attention': False,
            'normalize': True,
            'license': 'MIT',
            'source_repo': 'MVRL/TTE',
            'upstream_url': 'https://github.com/mvrl/TTE',
            'sha256': (
                'fa3f97476c3c6ee705dc45598c2e4977f31f585b8ba057c93339ca84415175f9'
            ),
        },
    )


def _strip_training_keys(state: dict[str, Tensor]) -> dict[str, Tensor]:
    """Remove training-only image register tensors from upstream TTE checkpoints."""
    return {k: v for k, v in state.items() if k not in _UPSTREAM_IMAGE_KEYS}


class TTELocationEncoder(LocationEncoder):
    """Tessellating the Earth (TTE) location encoder.

    Accepts universal ``(lon, lat)`` WGS84 degrees and internally swaps to the
    ``(lat, lon)`` ordering expected by pretrained TTE weights. Produces 512-d
    L2-normalized embeddings.
    """

    name = 'tte'

    def __init__(
        self,
        weights: TTE_Weights | str | None = TTE_Weights.DEFAULT,
        *,
        checkpoint_path: Path | None = None,
        n_sites: int = 4096,
        site_embed_dim: int = 384,
        output_dim: int = 512,
        hidden_dim: int = 512,
        n_reslayers: int = 2,
        n_registers: int = 64,
        register_fixed_gate: float = 0.5,
        use_cosine_attention: bool = False,
        normalize: bool = True,
        device: str | torch.device = 'cpu',
        batch_size: int = 8192,
    ) -> None:
        """Initialize a new TTELocationEncoder instance.

        Args:
            weights: Pretrained :class:`TTE_Weights` member, name, or ``None``.
            checkpoint_path: Optional local ``.safetensors`` checkpoint path.
            n_sites: Number of learnable Voronoi sites.
            site_embed_dim: Per-site embedding dimension.
            output_dim: Output embedding dimension.
            hidden_dim: Residual MLP hidden width.
            n_reslayers: Number of residual layers in the MLP head.
            n_registers: Number of semantic register tokens.
            register_fixed_gate: Fixed skip-connection blend weight.
            use_cosine_attention: Whether to use cosine register attention.
            normalize: Whether to L2-normalize output embeddings.
            device: Target computation device.
            batch_size: Batch size used by :meth:`encode`.
        """
        super().__init__(device=device, batch_size=batch_size)
        self.embed_dim = output_dim
        self.model = TTENet(
            n_sites=n_sites,
            site_embed_dim=site_embed_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            n_reslayers=n_reslayers,
            n_registers=n_registers,
            register_fixed_gate=register_fixed_gate,
            use_cosine_attention=use_cosine_attention,
            normalize=normalize,
        )

        ckpt_path: Path | None = checkpoint_path
        if ckpt_path is None and weights is not None:
            resolved_weights = TTE_Weights.verify(weights)
            ckpt_path = resolved_weights.download()
        if ckpt_path is not None:
            state = _strip_training_keys(load_safetensors(ckpt_path, device='cpu'))
            self.model.load_state_dict(state, strict=True)

        self.model.to(self.device)
        self.model.eval()

    def forward(
        self, coords: Tensor, year: Tensor | None = None, month: Tensor | None = None
    ) -> Tensor:
        """Encode geographic ``(lon, lat)`` coordinates into TTE embeddings.

        Args:
            coords: Coordinate tensor of shape ``(B, 2)`` in ``(lon, lat)`` degrees.
            year: Unused temporal input kept for :class:`LocationEncoder` parity.
            month: Unused temporal input kept for :class:`LocationEncoder` parity.

        Returns:
            Float32 embedding tensor of shape ``(B, output_dim)``.
        """
        return self.model(coords.flip(-1))


def tte(
    weights: TTE_Weights | str | None = TTE_Weights.DEFAULT, **kwargs: Any
) -> TTELocationEncoder:
    """Create a Tessellating the Earth (TTE) location encoder.

    Args:
        weights: Pretrained :class:`TTE_Weights` member, name, or ``None``.
        **kwargs: Additional keyword arguments passed to :class:`TTELocationEncoder`.

    Returns:
        Configured :class:`TTELocationEncoder` instance.
    """
    return TTELocationEncoder(weights=weights, **kwargs)


__all__ = (
    'SemanticRegisterAttention',
    'TTELocationEncoder',
    'TTENet',
    'TTE_Weights',
    'VoronoiLocationEncoder',
    'tte',
)
