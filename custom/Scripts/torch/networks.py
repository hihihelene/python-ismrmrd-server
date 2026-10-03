"""
from https://github.com/voxelmorph/voxelmorph/pull/571/changes
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Literal

import neurite as ne
import torch
import torch.nn.functional as F
import voxelmorph as vxm
from torch import nn

from .hyperconv import (
    HyperConv2dFromDense,
    HyperConv3dFromDense,
)


class HyperConvBlock(nn.Module):
    def __init__(
        self,
        ndim,
        in_channels,
        out_channels,
        hyp_num_outputs,
        activation=nn.LeakyReLU,
        padding_mode="zeros",
    ):
        super().__init__()

        if ndim == 2:
            Conv = HyperConv2dFromDense
        elif ndim == 3:
            Conv = HyperConv3dFromDense
        else:
            raise ValueError(f"Unsupported ndim={ndim}; expected 2 or 3.")

        self.conv = Conv(
            hyp_inputs=hyp_num_outputs,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
            padding_mode=padding_mode,
        )

        self.activation = activation()

    def forward(
        self,
        x: torch.Tensor,
        hyp_tensor: torch.Tensor,
    ) -> torch.Tensor:
        return self.activation(self.conv(x, hyp_tensor))


class HyperBasicUNet(nn.Module):
    """
    Hypernetwork-conditioned U-Net.

    The convolution kernels of the U-Net are generated from
    `hyp_tensor` on a per-sample basis.

    The network itself produces feature maps. The final flow
    projection is intentionally kept outside this class, matching
    the current VxmPairwise architecture.
    """

    def __init__(
        self,
        ndim,
        in_channels,
        out_channels,
        nb_features=(16, 16, 16, 16, 16),
        hyp_num_outputs=None,
        activations=nn.LeakyReLU,
        final_activation=None,
        padding_mode="zeros",
        upsample_mode="nearest",
    ):
        super().__init__()

        if ndim not in (2, 3):
            raise ValueError(f"Unsupported ndim={ndim}; expected 2 or 3.")

        if hyp_num_outputs is None:
            raise ValueError("hyp_num_outputs must be specified.")

        if len(nb_features) < 2:
            raise ValueError("HyperBasicUNet requires at least two feature levels.")

        self.ndim = ndim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.nb_features = tuple(nb_features)
        self.hyp_num_outputs = hyp_num_outputs
        self.upsample_mode = upsample_mode

        # ---------------------------------------------------------
        # Encoder
        # ---------------------------------------------------------

        self.encoder = nn.ModuleList()

        current_channels = in_channels

        for features in self.nb_features:
            self.encoder.append(
                HyperConvBlock(
                    ndim=ndim,
                    in_channels=current_channels,
                    out_channels=features,
                    hyp_num_outputs=hyp_num_outputs,
                    activation=activations,
                    padding_mode=padding_mode,
                )
            )

            current_channels = features

        # ---------------------------------------------------------
        # Downsampling
        # ---------------------------------------------------------

        if ndim == 2:
            self.pool = nn.MaxPool2d(
                kernel_size=2,
                stride=2,
            )
        else:
            self.pool = nn.MaxPool3d(
                kernel_size=2,
                stride=2,
            )

        # ---------------------------------------------------------
        # Bottleneck
        # ---------------------------------------------------------

        bottleneck_channels = self.nb_features[-1]

        self.bottleneck = HyperConvBlock(
            ndim=ndim,
            in_channels=bottleneck_channels,
            out_channels=bottleneck_channels,
            hyp_num_outputs=hyp_num_outputs,
            activation=activations,
            padding_mode=padding_mode,
        )

        # ---------------------------------------------------------
        # Decoder
        # ---------------------------------------------------------

        self.decoder = nn.ModuleList()

        current_channels = bottleneck_channels

        # One decoder stage for every encoder skip except
        # the deepest encoder level.
        for skip_channels in reversed(self.nb_features[:-1]):
            self.decoder.append(
                HyperConvBlock(
                    ndim=ndim,
                    in_channels=current_channels + skip_channels,
                    out_channels=skip_channels,
                    hyp_num_outputs=hyp_num_outputs,
                    activation=activations,
                    padding_mode=padding_mode,
                )
            )

            current_channels = skip_channels

        # ---------------------------------------------------------
        # Output projection
        #
        # This is still hypernetwork-controlled.
        # The separate VxmPairwise flow layer comes afterwards.
        # ---------------------------------------------------------

        if ndim == 2:
            Conv = HyperConv2dFromDense
        else:
            Conv = HyperConv3dFromDense

        self.out_layer = Conv(
            hyp_inputs=hyp_num_outputs,
            in_channels=current_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
            padding_mode=padding_mode,
        )

        self.final_activation = self._make_activation(final_activation)

    @staticmethod
    def _make_activation(activation):
        if activation is None:
            return None

        if isinstance(activation, nn.Module):
            return activation

        if isinstance(activation, str):
            return getattr(nn, activation)()

        if isinstance(activation, type):
            return activation()

        if callable(activation):
            return activation()

        raise TypeError(f"Unsupported final_activation: {activation!r}")

    def _upsample(self, x, target_size):
        if self.upsample_mode == "nearest":
            return F.interpolate(
                x,
                size=target_size,
                mode="nearest",
            )

        if self.ndim == 2:
            mode = "bilinear"
        else:
            mode = "trilinear"

        return F.interpolate(
            x,
            size=target_size,
            mode=mode,
            align_corners=False,
        )

    def forward(self, x, hyp_tensor):

        if hyp_tensor.ndim != 2:
            raise ValueError("hyp_tensor must have shape [B, H].")

        if hyp_tensor.shape[0] != x.shape[0]:
            raise ValueError("Batch size mismatch between input and hyperparameters.")

        skips = []

        # ---------------------------------------------------------
        # Encoder
        # ---------------------------------------------------------

        for level, block in enumerate(self.encoder):
            x = block(x, hyp_tensor)

            skips.append(x)

            # No pooling after the deepest level.
            if level < len(self.encoder) - 1:
                x = self.pool(x)

        # ---------------------------------------------------------
        # Bottleneck
        # ---------------------------------------------------------

        x = self.bottleneck(x, hyp_tensor)

        # ---------------------------------------------------------
        # Decoder
        # ---------------------------------------------------------

        # Skip tensors excluding the deepest encoder feature map.
        skips = skips[:-1]

        for block, skip in zip(
            self.decoder,
            reversed(skips),
        ):
            x = self._upsample(
                x,
                target_size=skip.shape[2:],
            )

            x = torch.cat(
                [x, skip],
                dim=1,
            )

            x = block(
                x,
                hyp_tensor,
            )

        # ---------------------------------------------------------
        # Hypernetwork-controlled output projection
        # ---------------------------------------------------------

        x = self.out_layer(
            x,
            hyp_tensor,
        )

        if self.final_activation is not None:
            x = self.final_activation(x)

        return x


class HyperVxmPairwise(nn.Module):
    """
    HyperMorph version of the current VoxelMorph VxmPairwise model.

    The Hypernetwork generates the convolution weights used by
    HyperBasicUNet. The final flow projection remains a normal
    Neurite ConvBlock, matching current VxmPairwise.
    """

    def __init__(
        self,
        ndim,
        source_channels,
        target_channels,
        nb_features=(16, 16, 16, 16, 16),
        nb_hyp_params=1,
        nb_hyp_layers=6,
        nb_hyp_units=128,
        activations=nn.LeakyReLU,
        final_activation=None,
        flow_initializer=1e-5,
        integration_steps=5,
        upsample_mode="nearest",
        padding_mode="zeros",
    ):
        super().__init__()

        self.ndim = ndim
        self.source_channels = source_channels
        self.target_channels = target_channels
        self.nb_features = tuple(nb_features)

        self.nb_hyp_params = nb_hyp_params
        self.nb_hyp_layers = nb_hyp_layers
        self.nb_hyp_units = nb_hyp_units

        self.integration_steps = integration_steps

        # ---------------------------------------------------------
        # Hypernetwork
        # ---------------------------------------------------------

        hyp_layers = []

        in_features = nb_hyp_params

        for _ in range(nb_hyp_layers):
            hyp_layers.append(
                nn.Linear(
                    in_features,
                    nb_hyp_units,
                )
            )
            hyp_layers.append(nn.LeakyReLU())

            in_features = nb_hyp_units

        self.hyp_model = nn.Sequential(*hyp_layers)

        # ---------------------------------------------------------
        # Hyper-conditioned U-Net
        # ---------------------------------------------------------

        self.model = HyperBasicUNet(
            ndim=ndim,
            in_channels=(source_channels + target_channels),
            out_channels=ndim,
            nb_features=nb_features,
            hyp_num_outputs=nb_hyp_units,
            activations=activations,
            final_activation=final_activation,
            padding_mode=padding_mode,
            upsample_mode=upsample_mode,
        )

        # ---------------------------------------------------------
        # IMPORTANT:
        #
        # The flow layer is NOT hyperized.
        #
        # This follows current VxmPairwise:
        #
        # BasicUNet -> ConvBlock(ndim, ndim, ndim)
        # ---------------------------------------------------------

        self.flow_layer = ne.nn.modules.ConvBlock(
            ndim,
            ndim,
            ndim,
        )

        # Match current VxmPairwise flow initialization.
        self._initialize_flow_layer(flow_initializer)

        # ---------------------------------------------------------
        # Integration
        # ---------------------------------------------------------

        if integration_steps > 0:
            self.velocity_field_integrator = vxm.nn.modules.IntegrateVelocityField(
                steps=integration_steps,
            )

        # ---------------------------------------------------------
        # Spatial transformer
        # ---------------------------------------------------------

        self.spatial_transformer = vxm.nn.modules.SpatialTransformer()

    def _initialize_flow_layer(
        self,
        std,
    ):
        """
        Match VxmPairwise's near-zero flow initialization.
        """

        conv = getattr(
            self.flow_layer,
            "conv0",
            None,
        )

        if conv is None:
            conv = getattr(
                self.flow_layer,
                "conv",
                None,
            )

        if conv is None:
            raise AttributeError(
                "Could not find convolution weights in Neurite ConvBlock."
            )

        nn.init.normal_(
            conv.weight,
            mean=0.0,
            std=std,
        )

        if conv.bias is not None:
            nn.init.zeros_(conv.bias)

    def forward(
        self,
        source,
        target,
        hyperparameters,
        return_warped_source=False,
        return_warped_target=False,
        return_field_type="displacement",
    ):
        if return_field_type not in (
            "displacement",
            "velocity",
            "svf",
        ):
            raise ValueError(
                "return_field_type must be one of 'displacement', 'velocity', or 'svf'."
            )

        if hyperparameters is None:
            raise ValueError("HyperVxmPairwise requires `hyperparameters`.")

        if hyperparameters.ndim == 1:
            hyperparameters = hyperparameters.unsqueeze(-1)

        if hyperparameters.ndim != 2:
            raise ValueError(
                "hyperparameters must have shape [B, H]. "
                f"Got {tuple(hyperparameters.shape)}."
            )

        if hyperparameters.shape[0] != source.shape[0]:
            raise ValueError("Batch size mismatch between source and hyperparameters.")

        hyperparameters = hyperparameters.to(
            device=source.device,
            dtype=source.dtype,
        )

        # ---------------------------------------------------------
        # Hypernetwork
        # ---------------------------------------------------------

        hyp_tensor = self.hyp_model(hyperparameters)

        # ---------------------------------------------------------
        # Pairwise input
        # ---------------------------------------------------------

        combined = torch.cat(
            [source, target],
            dim=1,
        )

        # ---------------------------------------------------------
        # Hyper-U-Net
        # ---------------------------------------------------------

        features = self.model(
            combined,
            hyp_tensor,
        )

        # ---------------------------------------------------------
        # Velocity field
        # ---------------------------------------------------------

        velocity = self.flow_layer(features)

        # ---------------------------------------------------------
        # Return raw velocity/SVF if requested.
        #
        # Current VxmPairwise treats velocity and SVF as the
        # unintegrated field.
        # ---------------------------------------------------------

        if return_field_type in (
            "velocity",
            "svf",
        ):
            field = velocity

        else:
            if self.integration_steps > 0:
                field = self.velocity_field_integrator(velocity)
            else:
                field = velocity

        outputs = [field]

        # ---------------------------------------------------------
        # Warp source
        # ---------------------------------------------------------

        if return_warped_source:
            warped_source = self.spatial_transformer(
                source,
                field,
            )
            outputs.append(warped_source)

        # ---------------------------------------------------------
        # Warp target
        # ---------------------------------------------------------

        if return_warped_target:
            warped_target = self.spatial_transformer(
                target,
                -field,
            )
            outputs.append(warped_target)

        if len(outputs) == 1:
            return outputs[0]

        return tuple(outputs)
