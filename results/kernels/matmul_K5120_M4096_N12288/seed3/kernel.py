import numpy as np
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.typing as nt
import neuronxcc.nki.isa as nisa
from neuronxcc.nki import trace
from neuronxcc.nki.language import par_dim

@nki.jit
def kernel(v1, v2):
    v3 = nl.ndarray((32, 128, 12288), dtype=np.float32, buffer=nl.shared_hbm)
    for inb in nl.affine_range(12):
        b = nl.ndarray((5, 8, nl.par_dim(128), 1024), dtype=np.float32,
                       name='b_tiles', buffer=nl.sbuf)
        for ikg in nl.affine_range(5):
            for ik in nl.affine_range(8):
                b[ikg, ik, nl.arange(128)[:, None], nl.arange(1024)[None, :]] = \
                    nl.load(v2[ik + 8 * ikg, nl.arange(128)[:, None],
                               nl.arange(1024)[None, :] + 1024 * inb],
                            dtype=np.float32)

        for im in nl.affine_range(32):
            a = nl.ndarray((5, 8, nl.par_dim(128), 128), dtype=np.float32,
                           name='a_tiles', buffer=nl.sbuf)
            for ikg in nl.affine_range(5):
                for ik in nl.affine_range(8):
                    a[ikg, ik, nl.arange(128)[:, None], nl.arange(128)[None, :]] = \
                        nl.load_transpose2d(
                            v1[im, nl.arange(128)[:, None], 8 * ikg + ik,
                               nl.arange(128)[None, :]], dtype=np.float32)
            acc = nl.zeros((2, nl.par_dim(128), 512), dtype=np.float32,
                           name='acc', buffer=nl.psum, lazy_initialization=True)
            for ino in nl.affine_range(2):
                for ikg in nl.affine_range(5):
                    for ik in nl.affine_range(8):
                        acc[ino, nl.arange(128)[:, None], nl.arange(512)[None, :]] += \
                            nisa.nc_matmul(
                                a[ikg, ik, nl.arange(128)[:, None], nl.arange(128)[None, :]],
                                b[ikg, ik, nl.arange(128)[:, None],
                                  512 * ino + nl.arange(512)[None, :]],
                                is_stationary_onezero=False,
                                is_moving_onezero=False, mask=None)
            nl.store(v3[im, nl.arange(128)[:, None],
                        nl.arange(512)[None, :] + 1024 * inb],
                     value=acc[0, nl.arange(128)[:, None],
                               nl.arange(512)[None, :]])
            nl.store(v3[im, nl.arange(128)[:, None],
                        nl.arange(512)[None, :] + 1024 * inb + 512],
                     value=acc[1, nl.arange(128)[:, None],
                               nl.arange(512)[None, :]])

    return v3
