"""
Adapted version of the batch normalization and instance normalization in PyTorch:
https://pytorch.org/docs/stable/_modules/torch/nn/modules/batchnorm.html
and:
https://github.com/ptrblck/pytorch_misc
to act on 6D Scale-Translation feature maps.
"""
import torch
from torch.nn import Module
from torch.nn import functional as F
from torch.nn.parameter import Parameter
from typing import Optional
from torch.nn import init
from torch import Tensor


class _NormBase(Module):
    """Common base of _InstanceNorm and _BatchNorm"""

    _version = 2
    __constants__ = ["track_running_stats", "momentum", "eps", "num_features", "affine"]
    num_features: int
    eps: float
    momentum: float
    affine: bool
    track_running_stats: bool

    # WARNING: weight and bias purposely not defined here.
    # See https://github.com/pytorch/pytorch/issues/39670

    def __init__(
            self,
            num_features: int,
            eps: float = 1e-5,
            momentum: float = 0.1,
            affine: bool = True,
            track_running_stats: bool = True,
            device=None,
            dtype=None
    ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(_NormBase, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        if self.affine:
            self.weight = Parameter(torch.empty(num_features, **factory_kwargs))
            self.bias = Parameter(torch.empty(num_features, **factory_kwargs))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features, **factory_kwargs))
            self.register_buffer('running_var', torch.ones(num_features, **factory_kwargs))
            self.running_mean: Optional[Tensor]
            self.running_var: Optional[Tensor]
            self.register_buffer('num_batches_tracked',
                                 torch.tensor(0, dtype=torch.long,
                                              **{k: v for k, v in factory_kwargs.items() if k != 'dtype'}))
        else:
            self.register_buffer("running_mean", None)
            self.register_buffer("running_var", None)
            self.register_buffer("num_batches_tracked", None)
        self.reset_parameters()

    def reset_running_stats(self) -> None:
        if self.track_running_stats:
            # running_mean/running_var/num_batches... are registered at runtime depending
            # if self.track_running_stats is on
            self.running_mean.zero_()  # type: ignore[union-attr]
            self.running_var.fill_(1)  # type: ignore[union-attr]
            self.num_batches_tracked.zero_()  # type: ignore[union-attr,operator]

    def reset_parameters(self) -> None:
        self.reset_running_stats()
        if self.affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)

    def _check_input_dim(self, input):
        raise NotImplementedError

    def extra_repr(self):
        return (
            "num_features={num_features}".format(**self.__dict__)
        )

    def _load_from_state_dict(
            self,
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
    ):
        version = local_metadata.get("version", None)

        if (version is None or version < 2) and self.track_running_stats:
            # at version 2: added num_batches_tracked buffer
            #               this should have a default value of 0
            num_batches_tracked_key = prefix + "num_batches_tracked"
            if num_batches_tracked_key not in state_dict:
                state_dict[num_batches_tracked_key] = torch.tensor(0, dtype=torch.long)

        super(_NormBase, self)._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )


class _BatchNorm(_NormBase):
    def __init__(
            self,
            num_features,
            eps=1e-5,
            momentum=0.1,
            affine=True,
            track_running_stats=True,
            device=None,
            dtype=None
    ):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(_BatchNorm, self).__init__(
            num_features, eps, momentum, affine, track_running_stats, **factory_kwargs
        )

    def forward(self, input: Tensor) -> Tensor:
        self._check_input_dim(input)

        # exponential_average_factor is set to self.momentum
        # (when it is available) only so that it gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:  # type: ignore[has-type]
                self.num_batches_tracked = self.num_batches_tracked + 1  # type: ignore[has-type]
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        r"""
        Decide whether the mini-batch stats should be used for normalization rather than the buffers.
        Mini-batch stats are used in training mode, and in eval mode when buffers are None.
        """
        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (self.running_var is None)

        r"""
        Buffers are only updated if they are to be tracked and we are in training mode. Thus they only need to be
        passed when the update should occur (i.e. in training mode when they are tracked), or when buffer stats are
        used for normalization (i.e. in eval mode when buffers are not None).
        """
        return F.batch_norm(
            input,
            # If buffers are not to be tracked, ensure that they won't be updated
            self.running_mean
            if not self.training or self.track_running_stats
            else None,
            self.running_var if not self.training or self.track_running_stats else None,
            self.weight,
            self.bias,
            bn_training,
            exponential_average_factor,
            self.eps,
        )


class _InstanceNorm(_NormBase):
    def __init__(
            self,
            num_features: int,
            eps: float = 1e-5,
            momentum: float = 0.1,
            affine: bool = False,
            track_running_stats: bool = False,
            device=None,
            dtype=None
    ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(_InstanceNorm, self).__init__(
            num_features, eps, momentum, affine, track_running_stats, **factory_kwargs)

    def _check_input_dim(self, input):
        raise NotImplementedError

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        version = local_metadata.get('version', None)
        # at version 1: removed running_mean and running_var when
        # track_running_stats=False (default)
        if version is None and not self.track_running_stats:
            running_stats_keys = []
            for name in ('running_mean', 'running_var'):
                key = prefix + name
                if key in state_dict:
                    running_stats_keys.append(key)
            if len(running_stats_keys) > 0:
                error_msgs.append(
                    'Unexpected running stats buffer(s) {names} for {klass} '
                    'with track_running_stats=False. If state_dict is a '
                    'checkpoint saved before 0.4.0, this may be expected '
                    'because {klass} does not track running stats by default '
                    'since 0.4.0. Please remove these keys from state_dict. If '
                    'the running stats are actually needed, instead set '
                    'track_running_stats=True in {klass} to enable them. See '
                    'the documentation of {klass} for details.'
                        .format(names=" and ".join('"{}"'.format(k) for k in running_stats_keys),
                                klass=self.__class__.__name__))
                for key in running_stats_keys:
                    state_dict.pop(key)

        super(_InstanceNorm, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, input: Tensor) -> Tensor:
        self._check_input_dim(input)
        return F.instance_norm(
            input, self.running_mean, self.running_var, self.weight, self.bias,
            self.training or not self.track_running_stats, self.momentum, self.eps)


class SEInstanceNorm(_InstanceNorm):

    def __init__(self, num_features: int, eps=1e-5, momentum=0.1,
                 affine=True, track_running_stats=True):
        super().__init__(num_features, eps, momentum, affine, track_running_stats)

    def _check_input_dim(self, input):
        if input.dim() != 6:
            raise ValueError('expected 6D input (got {}D input)'.format(input.dim()))

    def forward(self, input):
        self._check_input_dim(input)

        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        # calculate running estimates
        if self.training:
            mean = input.mean([2, 3, 4, 5])
            # use biased var in train
            var = input.var([2, 3, 4, 5], unbiased=False)
            with torch.no_grad():
                self.running_mean = exponential_average_factor * mean.mean(dim=0) \
                                    + (1 - exponential_average_factor) * self.running_mean
                # update running_var with unbiased var
                self.running_var = exponential_average_factor * var.mean(dim=0) \
                                   + (1 - exponential_average_factor) * self.running_var
        else:
            if self.track_running_stats:
                mean = self.running_mean[None, :]
                var = self.running_var[None, :]
            else:
                mean = input.mean([2, 3, 4, 5])
                var = input.mean([2, 3, 4, 5])

        input = (input - mean[:, :, None, None, None, None]) / (
            torch.sqrt(var[:, :, None, None, None, None] + self.eps))
        if self.affine:
            input = input * self.weight[None, :, None, None, None, None] + self.bias[None, :, None, None, None, None]

        return input


class SEBatchNorm(_BatchNorm):
    def __init__(self, num_features, eps=1e-5, momentum=0.1,
                 affine=True, track_running_stats=True):
        super(SEBatchNorm, self).__init__(num_features, eps, momentum, affine, track_running_stats)

    def _check_input_dim(self, input):
        if input.dim() != 6:
            raise ValueError('expected 6D input (got {}D input)'.format(input.dim()))

    def forward(self, input):
        self._check_input_dim(input)

        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        # calculate running estimates
        if self.training:
            mean = input.mean([0, 2, 3, 4, 5])
            # use biased var in train
            var = input.var([0, 2, 3, 4, 5], unbiased=False)
            if self.track_running_stats:
                n = input.numel() / input.size(1)
                if n == 1:
                    with torch.no_grad():
                        self.running_mean = exponential_average_factor * mean \
                                            + (1 - exponential_average_factor) * self.running_mean
                        # update running_var with unbiased var
                        self.running_var = exponential_average_factor * var * \
                                           + (1 - exponential_average_factor) * self.running_var
                else:
                    with torch.no_grad():
                        self.running_mean = exponential_average_factor * mean \
                                            + (1 - exponential_average_factor) * self.running_mean
                        # update running_var with unbiased var
                        self.running_var = exponential_average_factor * var * n / (n - 1) \
                                           + (1 - exponential_average_factor) * self.running_var
        else:
            if self.track_running_stats:
                mean = self.running_mean
                var = self.running_var
            else:
                mean = input.mean([0, 2, 3, 4, 5])
                var = input.var([0, 2, 3, 4, 5])

        input = (input - mean[None, :, None, None, None, None]) / (
            torch.sqrt(var[None, :, None, None, None, None] + self.eps))
        if self.affine:
            input = input * self.weight[None, :, None, None, None, None] + self.bias[None, :, None, None, None, None]

        return input
