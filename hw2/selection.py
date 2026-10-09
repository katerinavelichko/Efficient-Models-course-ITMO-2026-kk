import numpy as np


def forgetting_selection_score(forgetting: np.ndarray, ever_correct: np.ndarray) -> np.ndarray:
    score = np.asarray(forgetting, dtype=np.float64).copy()
    ever = np.asarray(ever_correct, dtype=bool)
    finite = score[ever] if ever.any() else score
    ceiling = float(finite.max()) + 1.0 if finite.size else 1.0
    score[~ever] = ceiling
    return score


def select_top(scores: np.ndarray, m: int, seed: int) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    rng = np.random.default_rng(seed)
    jitter = rng.random(scores.shape[0])
    order = np.lexsort((jitter, -scores))
    return order[:m]


def select_top_excluding(scores: np.ndarray, excluded, k: int, seed: int) -> np.ndarray:
    masked = np.asarray(scores, dtype=np.float64).copy()
    masked[list(excluded)] = -np.inf
    return select_top(masked, k, seed)


def random_subset(n: int, m: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.choice(n, size=m, replace=False)


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.shape[0], dtype=np.float64)
    sorted_vals = values[order]
    i = 0
    n = values.shape[0]
    while i < n:
        j = i
        while j + 1 < n and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x = x - x.mean()
    y = y - y.mean()
    denom = np.sqrt((x * x).sum() * (y * y).sum())
    if denom == 0:
        return float("nan")
    return float((x * y).sum() / denom)


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    return pearson(_rankdata(np.asarray(x)), _rankdata(np.asarray(y)))


def spearman_masked(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 3:
        return float("nan")
    return spearman(x[mask], y[mask])


def kcenter_scores(features: np.ndarray, start: np.ndarray) -> np.ndarray:
    feats = np.ascontiguousarray(features, dtype=np.float32)
    n = feats.shape[0]
    selected = [int(i) for i in start]
    min_dist = np.full(n, np.inf, dtype=np.float32)
    for j in selected:
        diff = feats - feats[j]
        dist = np.sqrt(np.einsum("ij,ij->i", diff, diff))
        np.minimum(min_dist, dist, out=min_dist)
    min_dist[selected] = -np.inf
    order = []
    for _ in range(n - len(selected)):
        u = int(np.argmax(min_dist))
        order.append(u)
        diff = feats - feats[u]
        dist = np.sqrt(np.einsum("ij,ij->i", diff, diff))
        np.minimum(min_dist, dist, out=min_dist)
        min_dist[u] = -np.inf
    scores = np.full(n, np.nan, dtype=np.float64)
    n_ranked = len(order)
    for rank, u in enumerate(order):
        scores[u] = float(n_ranked - rank)
    return scores
