import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn import init
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.utils import _ntuple


# TODO Check!
def _get_activation(act):
    """Create an activation module from None, string, class, or instance."""
    if act is None:
        return None

    if isinstance(act, nn.Module):
        return act

    if isinstance(act, str):
        return getattr(nn, act)()

    if isinstance(act, type) and issubclass(act, nn.Module):
        return act()

    raise TypeError(f"Unsupported activation specification: {act!r}")


class HyperWeight(nn.Module):
    """
    Dense mapping from a hypernetwork embedding to convolution weights.

    Input:
        [B, hyp_inputs]

    Output:
        [B, *target_shape]
    """

    def __init__(
        self,
        nb_hyp_features,
        target_shape,
        use_bias,
        activation,
        device=None,
        dtype=None,
    ):
        super().__init__()

        factory_kwargs = {
            "device": device,
            "dtype": dtype,
        }

        units = math.prod(target_shape)

        self.kernel = nn.Parameter(
            torch.empty(
                nb_hyp_features,
                units,
                **factory_kwargs,
            )
        )

        self.bias = (
            nn.Parameter(
                torch.empty(
                    units,
                    **factory_kwargs,
                )
            )
            if use_bias
            else None
        )

        self.activation = activation
        self.target_shape = tuple(target_shape)

        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(
            self.kernel,
            a=math.sqrt(5),
        )

        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.kernel)

            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                init.uniform_(
                    self.bias,
                    -bound,
                    bound,
                )

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.ndim != 2:
            raise ValueError(
                "HyperWeight expects a 2D tensor "
                f"[batch, features], got {tuple(inputs.shape)}"
            )

        if inputs.is_sparse:
            outputs = torch.sparse.mm(
                inputs,
                self.kernel,
            )
        else:
            outputs = inputs @ self.kernel

        if self.bias is not None:
            outputs = outputs + self.bias

        if self.activation is not None:
            outputs = self.activation(outputs)

        return outputs.reshape(
            inputs.shape[0],
            *self.target_shape,
        )


class HyperConvFromDense(_ConvNd):
    """
    N-dimensional convolution whose kernel and bias are generated
    by a hypernetwork.

    Forward:
        inputs:    [B, Cin, ...]
        hyp_tensor:[B, H]

    The hypernetwork generates a separate convolution kernel for
    every sample in the batch.
    """

    def __init__(
        self,
        spatial_dims,
        hyp_inputs,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
        hyperkernel_use_bias=True,
        hyperbias_use_bias=True,
        hyperkernel_activation=None,
        hyperbias_activation=None,
        device=None,
        dtype=None,
    ):
        if padding == "causal":
            raise ValueError("Causal padding is not supported for HyperConv")

        if groups != 1:
            raise ValueError("HyperConvFromDense currently supports groups=1 only.")

        nt = _ntuple(spatial_dims)

        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=nt(kernel_size),
            stride=nt(stride),
            padding=(padding if isinstance(padding, str) else nt(padding)),
            dilation=nt(dilation),
            transposed=False,
            output_padding=nt(0),
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype,
        )

        # The actual convolution weights are generated dynamically
        # from hyp_tensor and therefore must not exist as trainable
        # ConvNd parameters.
        self.weight.requires_grad = False
        self.weight = None

        if bias:
            assert self.bias is not None
            self.bias.requires_grad = False

        self.bias = None

        self.spatial_dims = spatial_dims
        self.use_bias = bias

        self._conv_fn = getattr(
            F,
            f"conv{spatial_dims}d",
        )

        kernel_shape = (
            out_channels,
            in_channels,
            *self.kernel_size,
        )

        self.hyperkernel = HyperWeight(
            hyp_inputs,
            kernel_shape,
            use_bias=hyperkernel_use_bias,
            activation=_get_activation(hyperkernel_activation),
            device=device,
            dtype=dtype,
        )

        self.hyperbias = None

        if bias:
            self.hyperbias = HyperWeight(
                hyp_inputs,
                (out_channels,),
                use_bias=hyperbias_use_bias,
                activation=_get_activation(hyperbias_activation),
                device=device,
                dtype=dtype,
            )

    def reset_parameters(self):
        # The convolution parameters are generated dynamically.
        pass

    def forward(
        self,
        inputs: Tensor,
        hyp_tensor: Tensor,
    ) -> Tensor:

        if inputs.ndim != self.spatial_dims + 2:
            raise ValueError(
                f"Expected {self.spatial_dims + 2}D input "
                f"[B,C,...], got {tuple(inputs.shape)}"
            )

        if hyp_tensor.ndim != 2:
            raise ValueError(
                f"hyp_tensor must have shape [B, H], got {tuple(hyp_tensor.shape)}"
            )

        batch_size = inputs.shape[0]

        if hyp_tensor.shape[0] != batch_size:
            raise ValueError(
                "Batch size mismatch between input and hyperparameters: "
                f"{batch_size} vs {hyp_tensor.shape[0]}"
            )

        weight = self.hyperkernel(hyp_tensor)

        bias = self.hyperbias(hyp_tensor) if self.hyperbias is not None else None

        if self.padding_mode != "zeros":
            inputs = F.pad(
                inputs,
                self._reversed_padding_repeated_twice,
                mode=self.padding_mode,
            )
            padding = _ntuple(self.spatial_dims)(0)
        else:
            padding = self.padding

        input_spatial = inputs.shape[2:]

        # [B, Cin, ...] -> [1, B*Cin, ...]
        inputs_grouped = inputs.reshape(
            1,
            batch_size * self.in_channels,
            *input_spatial,
        )

        # [B, Cout, Cin, ...] -> [B*Cout, Cin, ...]
        weight_grouped = weight.reshape(
            batch_size * self.out_channels,
            self.in_channels,
            *self.kernel_size,
        )

        if bias is not None:
            bias_grouped = bias.reshape(
                batch_size * self.out_channels,
            )
        else:
            bias_grouped = None

        outputs = self._conv_fn(
            inputs_grouped,
            weight_grouped,
            bias_grouped,
            self.stride,
            padding,
            self.dilation,
            batch_size,
        )

        return outputs.reshape(
            batch_size,
            self.out_channels,
            *outputs.shape[2:],
        )


class HyperConv2dFromDense(HyperConvFromDense):
    """2D hyper-convolution generated from a dense embedding."""

    def __init__(
        self,
        hyp_inputs,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
        **kwargs,
    ):
        super().__init__(
            spatial_dims=2,
            hyp_inputs=hyp_inputs,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            **kwargs,
        )


class HyperConv3dFromDense(HyperConvFromDense):
    """3D hyper-convolution generated from a dense embedding."""

    def __init__(
        self,
        hyp_inputs,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
        **kwargs,
    ):
        super().__init__(
            spatial_dims=3,
            hyp_inputs=hyp_inputs,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            **kwargs,
        )
