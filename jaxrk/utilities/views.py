import jax.numpy as np

def tile_view(inp, reps):
    return np.broadcast_to(inp.ravel(), (reps, inp.size)).reshape((reps * inp.shape[0], inp.shape[1]))

