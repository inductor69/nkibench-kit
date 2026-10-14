import numpy as np
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa

@nki.jit
def kernel(v1, v2):
    out = nl.ndarray((32, 128, 12288), dtype=np.float32, buffer=nl.shared_hbm)
    rhs = nl.ndarray((12, 5, 8, nl.par_dim(128), 1024), dtype=np.float32,
                     buffer=nl.sbuf)
    lhs = nl.ndarray((12, 32, 5, 8, nl.par_dim(128), 128), dtype=np.float32,
                     buffer=nl.sbuf)
    acc = nl.zeros((12, 32, 2, nl.par_dim(128), 512), dtype=np.float32,
                   buffer=nl.psum, lazy_initialization=True)

    for n0 in nl.affine_range(12):
        for k0 in nl.affine_range(5):
            for k1 in nl.affine_range(8):
                rhs[n0, k0, k1, nl.arange(128)[:, None], nl.arange(1024)[None, :]] = nl.load(
                    v2[k1 + 8 * k0, nl.arange(128)[:, None],
                       1024 * n0 + nl.arange(1024)[None, :]],
                    dtype=np.float32, mask=None)

        for m0 in nl.affine_range(32):
            for k0 in nl.affine_range(5):
                for k1 in nl.affine_range(8):
                    lhs[n0, m0, k0, k1, nl.arange(128)[:, None],
                        nl.arange(128)[None, :]] = nl.load_transpose2d(
                            v1[m0, nl.arange(128)[:, None],
                               8 * k0 + k1, nl.arange(128)[None, :]],
                            dtype=np.float32, mask=None)
                    for n1 in nl.affine_range(2):
                        acc[n0, m0, n1, nl.arange(128)[:, None], nl.arange(512)[None, :]] += nisa.nc_matmul(
                            lhs[n0, m0, k0, k1, nl.arange(128)[:, None],
                                nl.arange(128)[None, :]],
                            rhs[n0, k0, k1, nl.arange(128)[:, None],
                                512 * n1 + nl.arange(512)[None, :]],
                            is_stationary_onezero=False,
                            is_moving_onezero=False,
                            mask=None)
            for n1 in nl.affine_range(2):
                nl.store(out[m0, nl.arange(128)[:, None],
                             1024 * n0 + 512 * n1 + nl.arange(512)[None, :]],
                         value=acc[n0, m0, n1, nl.arange(128)[:, None],
                                   nl.arange(512)[None, :]], mask=None)
    return out
