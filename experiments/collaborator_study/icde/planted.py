"""Disjoint class-mixture sampling with balanced planted groups within domains."""
import numpy as np


def allocate(labels, classes, clients, samples, domain, domains, seed):
    if classes % 2 or clients % 2 or clients < 2:
        raise ValueError('Use an even class count and an even number of clients per domain.')
    rng = np.random.default_rng(seed)
    pools = [rng.permutation(np.flatnonzero(labels == c)) for c in range(classes)]
    # Fixed quantity skew makes the equal-domain FedAvg control nontrivial.
    scale = np.linspace(.5, 1.5, domains)[domain]
    unit = min(int(samples * scale) // (5 * classes),
               min(len(x) for x in pools) // (5 * clients))
    if unit < 1:
        raise ValueError('Insufficient samples for disjoint 70/30 profiles with every class present.')
    groups = rng.permutation(np.tile([0, 1], clients // 2))
    offsets = np.zeros(classes, dtype=int)
    shards = []
    for group in groups:
        shard = []
        for c in range(classes):
            n = (7 if (c < classes // 2) == (group == 0) else 3) * unit
            shard.extend(pools[c][offsets[c]:offsets[c]+n].tolist())
            offsets[c] += n
        rng.shuffle(shard)
        shards.append(shard)
    return shards, groups


def domain_balanced_weight(client, clients):
    total = sum(len(x['train']) for x in clients if x['domain'] == client['domain'])
    return len(client['train']) / total


def allocate_finite(labels, classes, clients, samples, domain, domains, seed, return_diagnostics=False):
    """Sample finite histograms from latent profiles, then take disjoint images.

    Per-domain sample size is bounded using the smallest class pool. If a
    multinomial draw still exhausts a pool, reject the whole domain allocation;
    never silently clip a class quota or sample duplicate images.
    """
    if classes % 2 or clients % 2 or samples < 10:
        raise ValueError('Even classes/clients and at least 10 target samples required')
    rng = np.random.default_rng(seed)
    pools = [rng.permutation(np.flatnonzero(labels == c)) for c in range(classes)]
    groups = rng.permutation(np.tile([0, 1], clients//2))
    scale = np.linspace(.5, 1.5, domains)[domain]
    # Half of the tightest class budget leaves room for finite-sample variation.
    size = min(int(samples*scale), int(min(map(len,pools))*classes/clients*.5))
    capacities = np.array(list(map(len, pools)))
    conservative_size = size
    if np.any(capacities == 0) or capacities.sum() < clients * 10:
        raise ValueError(f'Domain {domain}: cannot build the requested planted benchmark. '
                         f'Class counts={capacities.tolist()}; need all classes present and '
                         f'at least {clients*10} images for {clients} clients. Check the dataset.')
    # A mean-capacity bound is not a feasibility proof. Small domains may still
    # admit finite draws; accept only a complete allocation that fits all pools.
    if size < 10:
        size = 10
    profiles = np.array([[ (.7 if (c < classes//2) == (g == 0) else .3)/(classes//2)
                          for c in range(classes)] for g in range(2)])
    for attempt in range(10000):
        counts = np.array([rng.multinomial(size,profiles[g]) for g in groups])
        if np.all(counts.sum(0) <= np.array(list(map(len,pools)))):
            break
    else:
        raise RuntimeError(f'Domain {domain}: 10000 full allocation attempts failed; '
                           f'class counts={capacities.tolist()}, samples/client={size}. '
                           'Do not duplicate images or drop classes to bypass this failure.')
    offsets = np.zeros(classes,dtype=int)
    shards = []
    for row in counts:
        ids = []
        for c,n in enumerate(row):
            ids.extend(pools[c][offsets[c]:offsets[c]+n].tolist())
            offsets[c] += n
        rng.shuffle(ids); shards.append(ids)
    diagnostic = dict(domain_index=domain, available_class_counts=capacities.tolist(),
                      samples_per_client=size, conservative_samples_per_client=conservative_size,
                      small_domain_fallback=conservative_size < 10,
                      rejected_draws=attempt, selected_class_counts=counts.sum(0).tolist(),
                      sampling='multinomial profiles conditioned on available class capacities')
    return (shards, groups, diagnostic) if return_diagnostics else (shards, groups)
