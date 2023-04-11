"""
Basis computation, adapted to 3D from https://arxiv.org/abs/1910.11093 data (with several additions and improvements)
"""
import numpy as np
import torch
from torch.nn.functional import pad
import math
from typing import Iterable


def hermite_poly(x, n):
    """
    Hermite polynomial of order n calculated at X

    :param x: np.array
    :param n: int >= 0
    :return: array of shape X.shape
    """
    coeff = [0] * n + [1]
    func = np.polynomial.hermite_e.hermeval(x, coeff)  # uses Hermite-E polynomials
    return func


def onescale_grid_hermite_gaussian(size: int, scale: float, max_order: int = None):
    """
    Hermite polynomials with Gaussian envelope
    Filter basis: f(x,y,z) = 1/σ^3 * H_n(x/σ) * H_m(y/σ) * H_j(z/σ) * exp(-(x^2 + y^2 + z^2)/(2*σ^2))

    :param size: size of kernel bases (size x size x size)
    :param scale: σ in the basis equation
    :param max_order: maximum order of Hermite polynomials, if None: max_order = size - 1
    :return: (max_order + 1)^3 kernel bases for different orders
    """
    if max_order is None:
        max_order = size - 1

    x = np.linspace(-(size // 2), size // 2, size)
    y = np.linspace(-(size // 2), size // 2, size)
    z = np.linspace(-(size // 2), size // 2, size)

    order_z, order_y, order_x = np.indices([max_order + 1, max_order + 1, max_order + 1])

    # gaussian envelope
    g = np.exp(-x ** 2 / (2 * scale ** 2)) / scale

    basis_x = [g * hermite_poly(x / scale, n) for n in order_x.ravel()]
    basis_y = [g * hermite_poly(y / scale, m) for m in order_y.ravel()]
    basis_z = [g * hermite_poly(z / scale, k) for k in order_z.ravel()]

    basis_x = torch.Tensor(np.stack(basis_x))
    basis_y = torch.Tensor(np.stack(basis_y))
    basis_z = torch.Tensor(np.stack(basis_z))

    basis = torch.einsum("bixy, bxjy, bxyk -> bijk", basis_x[:, :, None, None], basis_y[:, None, :, None],
                         basis_z[:, None, None, :])

    return basis


def multiscale_hermite_gaussian(size: int, base_scale: float, max_order: int = 4, mult: float = 1.2,
                                num_funcs: int = None):
    """
    Kernel basis of Hermite polynomials with Gaussian Envelope.
    The maximum order is shared between functions. More functions are added
    by decreasing the scale.

    :param size: size of kernel bases (size x size x size)
    :param base_scale: base scale σ
    :param max_order: maximum order of Hermite polynomials, shared between different scales
    :param mult: effective scales are base_scale / (mult ** n)
    :param num_funcs: number of generated kernel bases, if None: num_funcs = size ** 2
    :return: num_funcs kernel bases for different scales and orders
    """
    num_funcs = num_funcs or size ** 3
    num_funcs_per_scale = ((max_order + 1) * (max_order + 2) * (max_order + 3)) // 6
    num_scales = math.ceil(num_funcs / num_funcs_per_scale)
    scales = [base_scale / (mult ** n) for n in range(num_scales)]
    if len(scales) > 1:
        print('Hermite scales:', scales)

    basis_x = []
    basis_y = []
    basis_z = []

    x = np.linspace(-(size // 2), size // 2, size)
    y = np.linspace(-(size // 2), size // 2, size)
    z = np.linspace(-(size // 2), size // 2, size)

    for scale in scales:
        g = np.exp(-x ** 2 / (2 * scale ** 2)) / scale

        order_z, order_y, order_x = np.indices([max_order + 1, max_order + 1, max_order + 1])
        mask = order_z + order_y + order_x <= max_order
        bx = [g * hermite_poly(x / scale, n) for n in order_x[mask]]
        by = [g * hermite_poly(y / scale, n) for n in order_y[mask]]
        bz = [g * hermite_poly(z / scale, n) for n in order_z[mask]]

        basis_x.extend(bx)
        basis_y.extend(by)
        basis_z.extend(bz)

    basis_x = torch.Tensor(np.stack(basis_x))[:num_funcs]
    basis_y = torch.Tensor(np.stack(basis_y))[:num_funcs]
    basis_z = torch.Tensor(np.stack(basis_z))[:num_funcs]

    basis = torch.einsum("bixy, bxjy, bxyk -> bijk", basis_x[:, :, None, None], basis_y[:, None, :, None],
                         basis_z[:, None, None, :])
    return basis


def steerable_basis(
        size: int,
        scales: Iterable[float],
        effective_size: int,
        mode: str = "onescale",
        padding: bool = True,
        mult=1.2,
        max_order=4):
    """
    Constructs a (possibly padded) kernel basis for convolutions.

    :param size: size of convolutional kernel
    :param scales: list of different scales
    :param effective_size: dictates number of functions in the kernel basis (see output)
    :param mode: 'onescale' or 'multiscale' -> method to use for computation of filters for a single scale
    :param padding: whether to use padding in the kernel or not
    :param mult: (only for multiscale basis)
    :param max_order: (only for multiscale basis)
    :return: kernel basis with dim = effective_size^3 x len(scales) x size x size x size
    """
    max_scale = max(scales)
    basis_tensors = []
    for scale in scales:
        if padding:
            if size % 2 == 1:
                size_before_pad = int(size * scale / max_scale) // 2 * 2 + 1
            else:
                # eventually apply negative padding in the end
                size_before_pad = int(size * scale / max_scale) // 2 * 2 + 2
        else:
            size_before_pad = size

        assert size_before_pad > 1, "Scaled kernel size must be > 1."
        if mode == "onescale":
            max_order = effective_size - 1
            basis = onescale_grid_hermite_gaussian(size_before_pad,
                                                   scale=scale,
                                                   max_order=max_order)
        elif mode == "multiscale":
            num_funcs = effective_size ** 3
            basis = multiscale_hermite_gaussian(size_before_pad,
                                                base_scale=scale,
                                                max_order=max_order,
                                                mult=mult,
                                                num_funcs=num_funcs)
        else:
            raise ValueError("Only 'onescale' and 'multiscale' kernel basis are supported.")
        basis = basis[None, :, :, :, :]
        pad_size = (size - size_before_pad) // 2
        basis = pad(basis, [pad_size] * 6)[0]
        basis_tensors.append(basis)
    return torch.stack(basis_tensors, 1)


def normalize_basis_by_min_scale(basis):
    """
    Normalization of basis by minimal scale

    :param basis: torch.Tensor
    :return: torch.Tensor; normalized basis
    """
    norm = basis.pow(2).sum([2, 3, 4], keepdim=True).sqrt()[:, [0]]
    return basis / norm
