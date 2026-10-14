import numpy as np
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa


@nki.jit
def kernel(v1):
    v2 = nl.ndarray((128, 32, 7168), dtype=np.float32, buffer=nl.shared_hbm)
    rows = nl.arange(128)[:, None]
    cols = nl.arange(1024)[None, :]
    for i1 in nl.affine_range(7):
        for i0 in nl.affine_range(32):
            nl.store(v2[rows, i0, cols + 1024 * i1],
                     value=nisa.activation(op=nl.silu,
                                           data=nl.load(v1[rows, i0, cols + 1024 * i1], dtype=np.float32),
                                           dtype=np.float32))
    return v2
