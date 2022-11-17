import torch
import numpy as np
import random


def worker_init_fn(wid):
    seed_sequence = np.random.SeedSequence(
        [torch.initial_seed(), wid]
    )

    to_seed = spawn_get(seed_sequence, 2, dtype=int)
    torch.random.manual_seed(to_seed)

    np_seed = spawn_get(seed_sequence, 2, dtype=np.ndarray)
    np.random.seed(np_seed)

    py_seed = spawn_get(seed_sequence, 2, dtype=int)
    random.seed(py_seed)


def spawn_get(seedseq, n_entropy, dtype):
    child = seedseq.spawn(1)[0]
    state = child.generate_state(n_entropy, dtype=np.uint32)

    if dtype == np.ndarray:
        return state
    elif dtype == int:
        state_as_int = 0
        for shift, s in enumerate(state):
            state_as_int = state_as_int + int((2 ** (32 * shift) * s))
        return state_as_int
    else:
        raise ValueError(f'not a valid dtype "{dtype}"')
