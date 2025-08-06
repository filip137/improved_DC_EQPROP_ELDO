import numpy as np
from typing import Optional, Union, Iterable


def _unwrap_object_array(obj):
    if isinstance(obj, np.ndarray) and obj.dtype == object and obj.shape == ():
        try:
            return obj.tolist()
        except Exception:  # fallback: leave as-is
            pass
    return obj


def load_weight_matrices(
    path: str,
    index: Optional[Union[int, slice, Iterable[int]]] = None,
    *,
    allow_pickle: bool = True,
):
    """
    Load weight_matrix_1 and weight_matrix_2 from a .npz file, optionally applying the same index to both.

    Returns
    -------
    (matrix1, matrix2)
    """
    with np.load(path, allow_pickle=allow_pickle) as data:
        missing = [k for k in ("weight_matrix_1", "weight_matrix_2") if k not in data]
        if missing:
            raise KeyError(
                f"Missing key(s) {missing!r} in {path}. Available keys: {list(data.keys())}"
            )

        w1 = _unwrap_object_array(data["weight_matrix_1"])
        w2 = _unwrap_object_array(data["weight_matrix_2"])

        if index is None:
            return w1, w2

        def apply_idx(obj):
            if isinstance(index, (int, slice)):
                return obj[index]
            if isinstance(index, Iterable):
                return [obj[i] for i in index]
            raise TypeError(f"Unsupported index type: {type(index)}")

        return apply_idx(w1), apply_idx(w2)



if __name__ == "__main__":
    path = "/home/filip/simulations/testing_plots/fet_FSST_0731/FSST_234103/weights.npz"
    w1, w2 = load_weight_matrices(path)
    print(w1)


