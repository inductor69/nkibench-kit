import numpy as np
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa

@nki.jit
def kernel(v1):
    out = nl.ndarray((128, 32, 7168), dtype=np.float32, buffer=nl.shared_hbm)
    p = nl.arange(128)[:, None]
    f = nl.arange(1024)[None, :]
    for i in nl.affine_range(32):
        for j in nl.affine_range(7):
            idx = 1024 * j + f
            x = nl.load(v1[p, i, idx], dtype=np.float32)
            y = nisa.activation(op=nl.silu, data=x, dtype=np.float32)
            nl.store(out[p, i, idx], value=y)
    return out
