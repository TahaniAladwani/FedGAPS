import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score


def predictive(confusions, names):
    all_c = np.sum(confusions, axis=0)
    support = all_c.sum(1)
    seen = support > 0
    recalls = np.divide(np.diag(all_c), support, out=np.zeros(len(support)), where=seen)
    domains = [float(np.trace(c) / c.sum()) if c.sum() else None for c in confusions]
    valid = [x for x in domains if x is not None]
    return dict(accuracy=float(np.trace(all_c) / all_c.sum()) if all_c.sum() else None,
        balanced_accuracy=float(recalls[seen].mean()) if seen.any() else None,
        worst_class_recall=float(recalls[seen].min()) if seen.any() else None,
        mean_domain_accuracy=float(np.mean(valid)) if valid else None,
        worst_domain_accuracy=min(valid) if valid else None,
        domain_accuracy_gap=max(valid)-min(valid) if valid else None,
        domain_accuracy=dict(zip(names, domains)), per_class_recall=[float(x) if s else None for x,s in zip(recalls,seen)],
        class_support=support.tolist(), confusion=all_c.tolist())


def clustering(membership, clients, distance=None):
    labels = membership.argmax(1)
    semantic = np.array([c['semantic_profile'] for c in clients])
    domain = np.array([c['domain'] for c in clients])
    valid = semantic >= 0
    output = {}
    for name, truth, mask in [('semantic', semantic, valid), ('domain', domain, np.ones(len(clients), bool))]:
        usable = len(np.unique(truth[mask])) > 1 and mask.sum() > 2
        output[name+'_ari'] = float(adjusted_rand_score(truth[mask], labels[mask])) if usable else None
        output[name+'_nmi'] = float(normalized_mutual_info_score(truth[mask], labels[mask])) if usable else None
        output[name+'_silhouette'] = None
        if distance is not None and usable and len(np.unique(truth[mask])) < mask.sum():
            d = np.maximum((distance + distance.T) / 2, 0)
            np.fill_diagonal(d, 0)
            output[name+'_silhouette'] = float(silhouette_score(d[np.ix_(mask, mask)], truth[mask], metric='precomputed'))
    a, b = output['semantic_silhouette'], output['domain_silhouette']
    output['semantic_minus_domain_silhouette'] = a-b if a is not None and b is not None else None
    counts = np.bincount(labels, minlength=membership.shape[1])
    proportions = counts / counts.sum()
    entropy = -np.sum(proportions * np.log(proportions.clip(1e-12)))
    output.update(hard_cluster_counts=counts.tolist(), soft_effective_clients=membership.sum(0).tolist(),
        effective_samples=(membership.T @ np.array([len(c['train']) for c in clients])).tolist(),
        cluster_allocation_balance=float(entropy/np.log(len(counts))) if len(counts)>1 else 1.,
        semantic_labels_evaluated=int(valid.sum()), semantic_label_kind='train_distribution_proxy')
    return output


class Monitor:
    def __init__(self, window, patience, min_delta, minimum):
        self.window, self.patience, self.min_delta, self.minimum = window, patience, min_delta, minimum
        self.history = []
        self.best = -1.
        self.best_round = 0
        self.bad = 0

    def update(self, score, round_id):
        self.history.append(float(score))
        current = float(np.mean(self.history[-self.window:]))
        improved = len(self.history) == 1 or current > self.best + self.min_delta
        if improved:
            self.best, self.best_round, self.bad = current, round_id, 0
        else:
            self.bad += 1
        return improved, round_id >= self.minimum and self.bad >= self.patience

    def export(self):
        return dict(best_round=self.best_round, stopped_round=len(self.history), best_validation_metric=self.best if self.history else None,
                    validation_history=self.history)
