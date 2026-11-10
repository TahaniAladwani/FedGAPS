import hashlib
import numpy as np
from sklearn.cluster import KMeans, SpectralClustering
from icde.engine import fuzzy_membership

def state_hash(state):
    h = hashlib.sha256()
    for name, value in sorted(state.items()):
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def assign(distance, vectors, method, k, seed, core, fuzziness=2., threshold=.1):
    if k == 1:
        return np.ones((len(distance), 1))
    if method == 'fuzzy':
        u, _ = fuzzy_membership(distance, k, fuzziness, core)
    else:
        if method == 'agglomerative':
            labels = core.hard_cluster(distance, k)
        elif method == 'spectral':
            positive = distance[distance > 0]
            scale = np.median(positive) if len(positive) else 1.
            affinity = np.exp(-distance**2 / (2 * scale**2))
            labels = SpectralClustering(k, affinity='precomputed', random_state=seed,
                                       n_init=20, assign_labels='kmeans').fit_predict(affinity)
        elif method == 'kmeans':
            labels = KMeans(k, n_init=20, random_state=seed).fit_predict(vectors)
        else:
            raise ValueError(method)
        u = np.eye(k)[labels]
    active = u >= threshold
    active[np.arange(len(u)), u.argmax(1)] = True
    u *= active
    return u / u.sum(1, keepdims=True)

