import numpy as np
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.typing as nt
import neuronxcc.nki.isa as nisa
from neuronxcc.nki import trace
from neuronxcc.nki.language import par_dim

@nki.jit
def kernel(v1, v2):
    import numpy as np
    import neuronxcc.nki as nki
    import neuronxcc.nki.language as nl
    import neuronxcc.nki.typing as nt
    import neuronxcc.nki.isa as nisa
    from neuronxcc.nki import trace
    from neuronxcc.nki.language import par_dim
    v3 = nl.ndarray((32, 128, 10944), dtype=np.float32, buffer=nl.shared_hbm)
    v4 = nl.ndarray((4, 16, nl.par_dim(128), 1024), dtype=np.float32, name='lhs_local_41', buffer=nl.sbuf)
    v5 = nl.ndarray((4, 11, 16, nl.par_dim(128), 1024), dtype=np.float32, name='', buffer=nl.sbuf)
    v6 = nl.zeros((4, 11, 8, 2, nl.par_dim(128), 512), dtype=np.float32, name='', buffer=nl.psum, lazy_initialization=True)
    for i0 in nl.affine_range(4):
        for i1 in nl.affine_range(16):
            v4[i0, i1, nl.arange(128)[:, None], nl.arange(1024)[None, :]] = nl.load(v1[nl.arange(128)[:, None], i1, 1024 * i0 + nl.arange(1024)[None, :]], dtype=np.float32, mask=None)
            ' end loop i1 '
        for i2 in nl.affine_range(10):
            for i3 in nl.affine_range(16):
                v5[i0, i2, i3, nl.arange(128)[:, None], nl.arange(1024)[None, :]] = nl.load(v2[nl.arange(128)[:, None], i3, 1024 * i2 + nl.arange(1024)[None, :]], dtype=np.float32, mask=None)
                ' end loop i3 '
            for i4 in nl.affine_range(8):
                for i5 in nl.affine_range(2):
                    for i6 in nl.affine_range(16):
                        v6[i0, i2, i4, i5, nl.arange(128)[:, None], nl.arange(512)[None, :]] += nisa.nc_matmul(v4[i0, i6, nl.arange(128)[:, None], nl.arange(128)[None, :] + 128 * i4], v5[i0, i2, i6, nl.arange(128)[:, None], 512 * i5 + nl.arange(512)[None, :]], is_stationary_onezero=False, is_moving_onezero=False)
                        ' end loop i6 '
                    nl.store(v3[8 * i0 + i4, nl.arange(128)[:, None], 1024 * i2 + 512 * i5 + nl.arange(512)[None, :]], value=nl.copy(v6[i0, i2, i4, i5, nl.arange(128)[:, None], nl.arange(512)[None, :]], dtype=np.float32))
                    ' end loop i5 '
                ' end loop i4 '
            ' end loop i2 '
        for i3 in nl.affine_range(16):
            v5[i0, 10, i3, nl.arange(128)[:, None], nl.arange(1024)[None, :]] = nl.load(v2[nl.arange(128)[:, None], i3, 10240 + nl.arange(1024)[None, :]], dtype=np.float32, mask=-10240 - 1 * nl.arange(1024)[None, :] + 10943 >= 0)
            ' end loop i3 '
        for i4 in nl.affine_range(8):
            for i6 in nl.affine_range(16):
                v6[i0, 10, i4, 0, nl.arange(128)[:, None], nl.arange(512)[None, :]] += nisa.nc_matmul(v4[i0, i6, nl.arange(128)[:, None], nl.arange(128)[None, :] + 128 * i4], v5[i0, 10, i6, nl.arange(128)[:, None], nl.arange(512)[None, :]], is_stationary_onezero=False, is_moving_onezero=False)
                ' end loop i6 '
            nl.store(v3[8 * i0 + i4, nl.arange(128)[:, None], 10240 + nl.arange(512)[None, :]], value=nl.copy(v6[i0, 10, i4, 0, nl.arange(128)[:, None], nl.arange(512)[None, :]], dtype=np.float32))
            for i6 in nl.affine_range(16):
                v6[i0, 10, i4, 1, nl.arange(128)[:, None], nl.arange(192)[None, :]] += nisa.nc_matmul(v4[i0, i6, nl.arange(128)[:, None], nl.arange(128)[None, :] + 128 * i4], v5[i0, 10, i6, nl.arange(128)[:, None], 512 + nl.arange(192)[None, :]], is_stationary_onezero=False, is_moving_onezero=False)
                ' end loop i6 '
            nl.store(v3[8 * i0 + i4, nl.arange(128)[:, None], 10752 + nl.arange(192)[None, :]], value=nl.copy(v6[i0, 10, i4, 1, nl.arange(128)[:, None], nl.arange(192)[None, :]], dtype=np.float32))
            ' end loop i4 '
        ' end loop i0 '
    return v3
