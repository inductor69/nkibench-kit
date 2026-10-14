import numpy as np
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa
import neuronxcc.nki.compiler as ncc
from neuronxcc.nki.compiler.backends.neuron.LexicalScopeDirective import auto_pipeline

@nki.jit
def kernel(v1):
    out = nl.ndarray((128, 32, 7168), dtype=np.float32, buffer=nl.shared_hbm)
    for i0 in nl.affine_range(32):
        x = nl.ndarray((7, nl.par_dim(128), 1024), dtype=np.float32, buffer=nl.sbuf)
        for i1 in nl.affine_range(7, directives=[ncc.multi_buffer(2), auto_pipeline(2)]):
            x[i1, nl.arange(128)[:, None], nl.arange(1024)[None, :]] = nl.load(
                v1[nl.arange(128)[:, None], i0, 1024 * i1 + nl.arange(1024)[None, :]],
                dtype=np.float32,
            )
            x[i1, nl.arange(128)[:, None], nl.arange(1024)[None, :]] = nisa.activation(
                op=nl.silu,
                data=x[i1, nl.arange(128)[:, None], nl.arange(1024)[None, :]],
                dtype=np.float32,
            )
            nl.store(
                out[nl.arange(128)[:, None], i0, 1024 * i1 + nl.arange(1024)[None, :]],
                value=x[i1, nl.arange(128)[:, None], nl.arange(1024)[None, :]],
            )
    return out
