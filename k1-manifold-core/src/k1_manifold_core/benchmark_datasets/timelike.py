import numpy as np


def make_timelike_dataset(n_seq, t_total, dim, rng, box=2.0, dt=0.08, noise=0.0):
    """Timelike geodesics: eta(v,v) > 0."""
    x = rng.uniform(-box, box, size=(n_seq, dim)).astype(np.float32)
    x[:, 0] = 0.0

    v = rng.normal(size=(n_seq, dim)).astype(np.float32) * 0.1
    spatial_norm = np.linalg.norm(v[:, 1:], axis=1, keepdims=True)
    v[:, 0] = 1.5 * spatial_norm[:, 0]

    traj = [x.copy()]
    for _ in range(t_total - 1):
        x = x + v * dt
        if noise > 0:
            x += noise * rng.normal(size=x.shape).astype(np.float32)
        traj.append(x.copy())

    return np.stack(traj, axis=1).astype(np.float32)
