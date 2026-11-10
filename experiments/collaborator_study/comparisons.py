"""Distances and diagnostics. Planted labels never enter a distance function."""
import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, mutual_info_score


def normalize(counts):
    values = np.asarray(counts,dtype=np.float64)
    if np.any(values < 0) or np.any(values.sum(1) <= 0):
        raise ValueError('Each client needs a nonempty nonnegative class histogram')
    return values/values.sum(1,keepdims=True)


def hellinger(counts):
    root = np.sqrt(normalize(counts))
    d = np.sqrt(np.maximum(((root[:,None]-root[None,:])**2).sum(2)/2,0))
    np.fill_diagonal(d,0)
    return d


def support_js(support, core):
    p = normalize(support)
    distance = np.zeros((len(p),len(p)),dtype=np.float64)
    for i in range(len(p)):
        for j in range(i):
            distance[i,j] = distance[j,i] = core.js_divergence(p[i],p[j])
    return distance


def distance_for(name, aligned, unaligned, core):
    if name == 'counts':
        return hellinger(aligned['counts'])
    if name == 'reliability_counts':
        # Exactly the existing JS support-distance term, without prototypes or coverage penalty.
        return support_js(aligned['support'], core)
    if name == 'unaligned':
        if unaligned is None:
            raise ValueError('The unaligned variant needs its separately trained no-graph encoder')
        return unaligned['distance'].copy()
    if name == 'aligned':
        return aligned['distance'].copy()
    raise ValueError(name)


def neighbor_recovery(distance, clients, k=3):
    """Precision@k among other-domain clients, using latent group truth.

    Average over tied boundary candidates instead of letting client IDs decide
    the reported precision. The fixed k is not silently reduced.
    """
    if any(c.get('planted_group',-1)<0 for c in clients):
        return None, []
    rows=[]
    for i,client in enumerate(clients):
        candidates=np.array([j for j,c in enumerate(clients) if c['domain']!=client['domain']])
        if len(candidates)<k:
            raise ValueError('Not enough other-domain clients for the predefined neighbor k')
        d=distance[i,candidates]
        cutoff=np.partition(d,k-1)[k-1]
        tie=np.isclose(d,cutoff,rtol=1e-10,atol=1e-12)
        nearer=(d<cutoff)&~tie
        correct=np.array([clients[j]['planted_group']==client['planted_group'] for j in candidates])
        score=(correct[nearer].sum()+(k-nearer.sum())*correct[tie].mean())/k
        rows.append(dict(client=client['id'],domain=client['domain'],planted_group=client['planted_group'],
                         neighbor_k=k,precision=float(score),boundary_ties=int(tie.sum()),
                         eligible_clients=len(candidates)))
    return float(np.mean([r['precision'] for r in rows])),rows


def discovery_metrics(u, distance, clients, k=3):
    labels=u.argmax(1)
    domain=[c['domain'] for c in clients]
    planted=all(c.get('planted_group',-1)>=0 for c in clients)
    group=[c.get('planted_group',-1) for c in clients]
    recovery,neighbors=neighbor_recovery(distance,clients,k)
    metrics=dict(planted_ari=float(adjusted_rand_score(group,labels)) if planted else None,
                 planted_nmi=float(normalized_mutual_info_score(group,labels)) if planted else None,
                 domain_mi_nats=float(mutual_info_score(domain,labels)),
                 domain_nmi=float(normalized_mutual_info_score(domain,labels)),
                 domain_ari=float(adjusted_rand_score(domain,labels)),
                 cross_domain_precision_at_k=recovery,neighbor_k=k,
                 planted_domain_mi_nats=float(mutual_info_score(domain,group)) if planted else None,
                 realized_clusters=int(len(np.unique(labels))),
                 active_models_per_client=float((u>0).sum(1).mean()),
                 soft_participation=u.mean(0).tolist(),
                 ground_truth='latent generating profile' if planted else 'no planted labels; recovery not evaluated')
    return metrics,neighbors


def packed_stats(stats, classes, feature_dim):
    n=len(stats)
    prototypes=np.zeros((n,classes,feature_dim),np.float32)
    counts=np.zeros((n,classes),np.int64)
    dispersion=np.zeros((n,classes),np.float32)
    support=np.zeros((n,classes),np.float64)
    for i,signature in enumerate(stats):
        for c,s in signature.items():
            prototypes[i,c]=s.prototype
            counts[i,c]=s.count
            dispersion[i,c]=s.dispersion
            support[i,c]=s.tempered_support
    return dict(prototypes=prototypes,counts=counts,dispersion=dispersion,support=support)


def summary_bytes(n, classes, features, kind):
    # Dense agreed class order: no class IDs; one int64 domain ID per client.
    if kind == 'full':
        return n*(classes*(features*4+8+4)+8)
    return n*(classes*8+8)  # int64 counts or float64 support.
