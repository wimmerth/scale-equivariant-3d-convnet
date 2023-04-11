"""
Scale-Equivariant Pooling Layers for the 3D setting.
"""
from torch.nn.functional import max_pool3d
from torch import nn
from typing import Optional


class SESProjection_H_Z3(nn.Module):

    def __init__(self,
                 projection_type: str = 'max'):
        """
        Projection of maximal activations, acts as a layer to make a network invariant to scale changes.
        This layer basically performs global pooling over the scaling group.
        """
        super(SESProjection_H_Z3, self).__init__()
        if projection_type not in ['max', 'avg']:
            raise ValueError("Only 'max' and 'avg' are supported as projection methods")
        self.projection_type = projection_type

    def forward(self, x):
        if self.projection_type == "max":
            max_, argmax_ = x.max(2)
            return max_
        else:
            return x.mean(2)

    def extra_repr(self):
        s = 'projection={projection_type}'
        return s.format(**self.__dict__)


class SESpatialMaxPool(nn.Module):
    __constants__ = ['kernel_size', 'stride', 'padding', 'dilation',
                     'return_indices', 'ceil_mode']
    return_indices: bool
    ceil_mode: bool

    def __init__(self, kernel_size: int, stride: Optional[int] = None,
                 padding: int = 0, dilation: int = 1,
                 return_indices: bool = False, ceil_mode: bool = False) -> None:
        """
        This layer performs spatial pooling over the 6-dimensional input (b * c * s * x * y * z).
        The required arguments are the same as known from torch.nn.MaxPool3d.
        """
        super(SESpatialMaxPool, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride if (stride is not None) else kernel_size
        self.padding = padding
        self.dilation = dilation
        self.return_indices = return_indices
        self.ceil_mode = ceil_mode

    def extra_repr(self) -> str:
        return 'kernel_size={kernel_size}, stride={stride}, padding={padding}' \
               ', dilation={dilation}, ceil_mode={ceil_mode}'.format(**self.__dict__)

    def forward(self, input):
        b, c, s, *xyz = input.shape
        input = input.view(b, -1, *xyz)
        input = max_pool3d(input, self.kernel_size, self.stride,
                           self.padding, self.dilation, self.ceil_mode,
                           self.return_indices)
        *_, x, y, z = input.shape
        input = input.view(b, c, s, x, y, z)
        return input
