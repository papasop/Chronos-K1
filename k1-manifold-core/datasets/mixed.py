import numpy as np

from .spacelike import make_spacelike_dataset
from .timelike import make_timelike_dataset


def make_mixed_dataset(n_seq, t_total, dim, rng, box=2.0, dt=0.08, noise=0.0, timelike_ratio=0.5):
    """Mixed dataset by concatenating timelike and spacelike trajectories."""
    n_t = int(round(n_seq * timelike_ratio))
    n_s = n_seq - n_t

    t = make_timelike_dataset(n_t, t_total, dim, rng, box=box, dt=dt, noise=noise)
    s = make_spacelike_dataset(n_s, t_total, dim, rng, box=box, dt=dt, noise=noise)

    out = np.concatenate([t, s], axis=0)
    rng.shuffle(out, axis=0)
    return out.astype(np.float32)
