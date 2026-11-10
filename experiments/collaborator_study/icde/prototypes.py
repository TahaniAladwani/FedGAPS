"""Prototype construction and numerically stable paper objectives."""
from collections import defaultdict
import math
import numpy as np
import torch
from torch.nn import functional as F
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


def finch_centres(features, block_size=1024):
    """FINCH last nontrivial partition, cosine first-neighbor graph.

    Exact blocked search avoids an N-by-N allocation. Shared-neighbor edges
    do not change connected components of the undirected first-neighbor graph.
    Means always use original samples, not unweighted means of means.
    """
    original = np.asarray(features, dtype=np.float32)
    if len(original) < 2:
        return original.copy()
    labels = np.arange(len(original))
    centres = original.copy()
    first = True
    while len(centres) > 1:
        z = centres / np.maximum(np.linalg.norm(centres, axis=1, keepdims=True), 1e-12)
        nearest = np.empty(len(z), dtype=np.int64)
        for start in range(0, len(z), block_size):
            end = min(start + block_size, len(z))
            similarities = z[start:end] @ z.T
            similarities[np.arange(end-start), np.arange(start, end)] = -np.inf
            nearest[start:end] = similarities.argmax(axis=1)
        graph = coo_matrix((np.ones(len(z)), (np.arange(len(z)), nearest)), shape=(len(z), len(z)))
        count, new = connected_components(graph, directed=False)
        if not first and (count == 1 or count >= len(centres)):
            break
        previous_count = len(centres)
        labels = new[labels]
        centres = np.zeros((count, original.shape[1]), dtype=np.float32)
        np.add.at(centres, labels, original)
        centres /= np.bincount(labels)[:, None]
        first = False
        if count == 1 or previous_count - count <= 1:
            break
    return centres


@torch.inference_mode()
def collect(model, loader, device, multi=False, normalized=False):
    model.eval()
    sums, counts, chunks = {}, defaultdict(int), defaultdict(list)
    for x, _, y, _, _ in loader:
        features = model.encoder(x.to(device, non_blocking=True)).float()
        if normalized:
            features = F.normalize(features, dim=1)
        features = features.cpu()
        for c in y.unique().tolist():
            values = features[y == c]
            counts[c] += len(values)
            sums[c] = sums.get(c, torch.zeros(values.shape[1])) + values.sum(0)
            if multi:
                chunks[c].append(values)
    if multi:
        return {c: torch.from_numpy(finch_centres(torch.cat(chunks[c]).numpy())) for c in sums}
    return {c: (sums[c] / counts[c]).unsqueeze(0) for c in sums}


def signatures(model, data, config, device):
    result = []
    for client in data.clients:
        means = collect(model, data.loader(client['id'], 'prototype'), device, normalized=True)
        stats = {}
        for c, mean in means.items():
            # For unit vectors, dispersion about the normalized mean is 1-||mean||.
            dispersion = max(0., 1 - float(mean.norm()))
            n = client['counts'][c]
            reliability = config.reliability_floor + (1-config.reliability_floor) * n / (n+config.tau_count) * math.exp(-config.tau_variance*dispersion)
            if config.method == 'ours_no_reliability':
                reliability = 1.
            stats[c] = data.core.ClassStat(F.normalize(mean[0], dim=0).numpy(), n, dispersion,
                n/sum(client['counts']), reliability, 0.)
        beta = 1. if config.method == 'ours_no_reliability' else config.tempering_beta
        total = sum(s.reliability * s.count**beta for s in stats.values())
        for s in stats.values():
            s.tempered_support = s.reliability * s.count**beta / total
        result.append(stats)
    return result


def aggregate(local, method, domains, config):
    grouped = defaultdict(list)
    for cid, prototypes in local.items():
        for c, values in prototypes.items():
            key = (c, domains[cid]) if method == 'feddap' else c
            grouped[key].extend(values)
    result = {}
    for key, values in grouped.items():
        values = torch.stack(values)
        if method in ('fedplvm', 'fpl'):
            result[key] = torch.from_numpy(finch_centres(values.numpy()))
        elif method == 'feddap':
            z = F.normalize(values, dim=1)
            weights = torch.softmax((z @ z.T).sum(1) / config.dap_aggregation_temperature, dim=0)
            result[key] = (weights[:, None] * values).sum(0, keepdim=True)
        else:
            result[key] = values.mean(0, keepdim=True)
    return result


def prepare_bank(bank, method, device):
    if not bank:
        return None
    values, labels, domains = [], [], []
    for key, tensor in sorted(bank.items()):
        c, d = key if method == 'feddap' else (key, -1)
        values.append(tensor)
        labels.extend([c] * len(tensor))
        domains.extend([d] * len(tensor))
    return (torch.cat(values).to(device), torch.tensor(labels, device=device), torch.tensor(domains, device=device))


def prototype_loss(features, labels, domain, bank, method, config):
    if bank is None:
        return features.sum() * 0
    values, classes, domains = bank
    positive = labels[:, None] == classes[None]
    present = positive.any(1)
    if method == 'fedproto':
        target = features.detach().clone()
        target[present] = values[positive[present].float().argmax(1)]
        return config.baseline_proto_weight * F.mse_loss(features, target)
    sim = F.normalize(features.float(), dim=1) @ F.normalize(values.float(), dim=1).T
    eligible = torch.ones_like(positive)
    extra = features.sum() * 0
    if method == 'feddap':
        same_domain = domains[None] == domain
        own = positive & same_domain
        extra = config.dap_intra_weight * ((1-sim) * own).sum(1).mean()
        eligible = ~same_domain.expand_as(positive)
        positive = positive & eligible
    if method == 'fedplvm':
        # ResNet's final ReLU gives nonnegative features; clamp protects round-off.
        sim = sim.clamp_min(1e-8).pow(config.plvm_power)
        # Paper Eq. 8: scalar L2 norm is absolute value, averaged over samples.
        extra = ((sim * positive).sum(1) - positive.sum(1)).abs()[present].sum() / len(labels)
    if method == 'fpl':
        target = (positive.float() @ values) / positive.sum(1, keepdim=True).clamp_min(1)
        extra = ((features - target).square().mean(1) * present).mean()
    valid = positive.any(1) & (eligible & ~positive).any(1)
    nce = features.sum() * 0
    if valid.any():
        logits = sim[valid] / config.temperature
        nce = (torch.logsumexp(logits.masked_fill(~eligible[valid], -torch.inf), 1)
              - torch.logsumexp(logits.masked_fill(~positive[valid], -torch.inf), 1)).sum() / len(labels)
    return extra + config.dap_cross_weight * nce if method == 'feddap' else config.baseline_proto_weight * (nce + extra)


def graph_bank(stats, data, model, config):
    core = data.core
    specs = [core.ClientSpec(c['id'], c['domain'], c['semantic_profile'], c['train'], c['counts']) for c in data.clients]
    return core.build_class_domain_graphs(stats, specs, model.feature_dim,
        0. if config.method == 'ours_no_graph' else config.graph_lambda,
        config.graph_eta, config.graph_temperature, config.graph_ridge)


def bank_to(bank, device):
    if bank is None:
        return None
    from dataclasses import fields, replace
    return replace(bank, **{f.name: getattr(bank, f.name).to(device) for f in fields(bank)})


def graph_nce(features, labels, bank, temperature):
    nodes = F.normalize(bank.smooth_nodes.float(), dim=-1)
    similarity = torch.einsum('bf,cdf->bcd', F.normalize(features.float(),dim=1), nodes) / temperature
    logits = (similarity*bank.mask[None]).sum(2) / bank.mask.sum(1).clamp_min(1)[None]
    observed = bank.mask.any(1)
    valid = observed[labels]
    if not valid.any():
        return features.sum()*0
    return F.cross_entropy(logits[valid].masked_fill(~observed[None],-torch.inf),labels[valid])


def ours_loss(features, second, labels, domains, bank, core, config, ramp):
    z = torch.cat((features, second))
    y = torch.cat((labels, labels))
    loss = features.sum() * 0
    if config.contrastive_weight:
        loss = loss + config.contrastive_weight * core.supervised_contrastive_loss(z, y, config.temperature)
    if config.consistency_weight:
        loss = loss + config.consistency_weight * (1-F.cosine_similarity(features, second)).mean()
    if bank is not None:
        if config.method != 'ours_no_graph':
            loss = loss + config.proto_weight * graph_nce(z, y, bank, config.temperature)
            loss = loss + config.alignment_weight * core.cross_domain_alignment_loss(z, y, torch.cat((domains, domains)), bank)
        centre, margin = core.global_centroid_losses(z, y, bank, config.margin)
        if config.method != 'ours_no_global_centre':
            loss = loss + config.centre_weight * centre
        if config.method != 'ours_no_margin':
            loss = loss + config.margin_weight * margin
    return ramp * loss


def rc_responsibilities(log_likelihood, membership, class_prior, labels):
    """FedRC Eq. 3, in log space; the label-prior correction is essential."""
    logits = log_likelihood + np.log(np.asarray(membership).clip(1e-30))
    logits -= np.log(class_prior[:, np.asarray(labels)].T.clip(1e-12))
    weights = np.exp(logits-logits.max(1,keepdims=True))
    return weights/weights.sum(1,keepdims=True)
