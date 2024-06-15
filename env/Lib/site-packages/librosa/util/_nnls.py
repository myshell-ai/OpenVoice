#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Non-negative least squares"""

# The scipy library provides an nnls solver, but it does
# not generalize efficiently to matrix-valued problems.
# We therefore provide an alternate solver here.
#
# The vectorized solver uses the L-BFGS-B over blocks of
# data to efficiently solve the constrained least-squares problem.

import numpy as np
import scipy.optimize
from .utils import MAX_MEM_BLOCK


__all__ = ["nnls"]


def _nnls_obj(x, shape, A, B):
    """Compute the objective and gradient for NNLS"""

    # Scipy's lbfgs flattens all arrays, so we first reshape
    # the iterate x
    x = x.reshape(shape)

    # Compute the difference matrix
    diff = np.einsum("mf,...ft->...mt", A, x, optimize=True) - B

    # Compute the objective value
    value = (1 / B.size) * 0.5 * np.sum(diff ** 2)

    # And the gradient
    grad = (1 / B.size) * np.einsum("mf,...mt->...ft", A, diff, optimize=True)

    # Flatten the gradient
    return value, grad.flatten()


def _nnls_lbfgs_block(A, B, x_init=None, **kwargs):
    """Solve the constrained problem over a single block

    Parameters
    ----------
    A : np.ndarray [shape=(m, d)]
        The basis matrix
    B : np.ndarray [shape=(m, N)]
        The regression targets
    x_init : np.ndarray [shape=(d, N)]
        An initial guess
    **kwargs
        Additional keyword arguments to `scipy.optimize.fmin_l_bfgs_b`

    Returns
    -------
    x : np.ndarray [shape=(d, N)]
        Non-negative matrix such that Ax ~= B
    """

    # If we don't have an initial point, start at the projected
    # least squares solution
    if x_init is None:
        x_init = np.einsum("fm,...mt->...ft", np.linalg.pinv(A), B, optimize=True)
        np.clip(x_init, 0, None, out=x_init)

    # Adapt the hessian approximation to the dimension of the problem
    kwargs.setdefault("m", A.shape[1])

    # Construct non-negative bounds
    bounds = [(0, None)] * x_init.size
    shape = x_init.shape

    # optimize
    x, obj_value, diagnostics = scipy.optimize.fmin_l_bfgs_b(
        _nnls_obj, x_init, args=(shape, A, B), bounds=bounds, **kwargs
    )
    # reshape the solution
    return x.reshape(shape)


def nnls(A, B, **kwargs):
    """Non-negative least squares.

    Given two matrices A and B, find a non-negative matrix X
    that minimizes the sum squared error::

        err(X) = sum_i,j ((AX)[i,j] - B[i, j])^2

    Parameters
    ----------
    A : np.ndarray [shape=(m, n)]
        The basis matrix
    B : np.ndarray [shape=(..., m, N)]
        The target array.  Additional leading dimensions are supported.
    **kwargs
        Additional keyword arguments to `scipy.optimize.fmin_l_bfgs_b`

    Returns
    -------
    X : np.ndarray [shape=(..., n, N), non-negative]
        A minimizing solution to ``|AX - B|^2``

    See Also
    --------
    scipy.optimize.nnls
    scipy.optimize.fmin_l_bfgs_b

    Examples
    --------
    Approximate a magnitude spectrum from its mel spectrogram

    >>> y, sr = librosa.load(librosa.ex('trumpet'), duration=3)
    >>> S = np.abs(librosa.stft(y, n_fft=2048))
    >>> M = librosa.feature.melspectrogram(S=S, sr=sr, power=1)
    >>> mel_basis = librosa.filters.mel(sr=sr, n_fft=2048, n_mels=M.shape[0])
    >>> S_recover = librosa.util.nnls(mel_basis, M)

    Plot the results

    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots(nrows=3, sharex=True, sharey=True)
    >>> librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max),
    ...                          y_axis='log', x_axis='time', ax=ax[2])
    >>> ax[2].set(title='Original spectrogram (1025 bins)')
    >>> ax[2].label_outer()
    >>> librosa.display.specshow(librosa.amplitude_to_db(M, ref=np.max),
    ...                          y_axis='mel', x_axis='time', ax=ax[0])
    >>> ax[0].set(title='Mel spectrogram (128 bins)')
    >>> ax[0].label_outer()
    >>> img = librosa.display.specshow(librosa.amplitude_to_db(S_recover, ref=np.max(S)),
    ...                          y_axis='log', x_axis='time', ax=ax[1])
    >>> ax[1].set(title='Reconstructed spectrogram (1025 bins)')
    >>> ax[1].label_outer()
    >>> fig.colorbar(img, ax=ax, format="%+2.0f dB")
    """

    # If B is a single vector, punt up to the scipy method
    if B.ndim == 1:
        return scipy.optimize.nnls(A, B)[0]

    n_columns = MAX_MEM_BLOCK // (np.prod(B.shape[:-1]) * A.itemsize)
    n_columns = max(n_columns, 1)

    # Process in blocks:
    if B.shape[-1] <= n_columns:
        return _nnls_lbfgs_block(A, B, **kwargs).astype(A.dtype)

    x = np.einsum("fm,...mt->...ft", np.linalg.pinv(A), B, optimize=True)
    np.clip(x, 0, None, out=x)
    x_init = x

    for bl_s in range(0, x.shape[-1], n_columns):
        bl_t = min(bl_s + n_columns, B.shape[-1])
        x[..., bl_s:bl_t] = _nnls_lbfgs_block(
            A, B[..., bl_s:bl_t], x_init=x_init[..., bl_s:bl_t], **kwargs
        )
    return x
