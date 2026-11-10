"""Single-device FL simulator with streaming aggregation and round recovery."""
import copy
import json
import math
import os
import platform
import shutil
import time
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .data import prepare
from .metrics import predictive, clustering, Monitor
from .model import Network
from .prototypes import (collect, aggregate, prepare_bank, prototype_loss, signatures,
                         graph_bank, bank_to, ours_loss, rc_responsibilities)
from .utils import Average, atomic_json, atomic_torch, code_hash, cpu_state, seed_all, tensor_bytes


def fuzzy_membership(distance, clusters, fuzziness, core, iterations=100):
    """Consistent squared-distance fuzzy medoids with distinct medoid indices."""
    hard = core.hard_cluster(distance, clusters)
    medoids = np.array([ids[np.argmin(distance[np.ix_(ids, ids)].sum(1))]
                       for ids in [np.flatnonzero(hard == k) for k in range(clusters)]])
    def memberships(m):
        d = distance[:, m]
        log_w = -2. / (fuzziness-1) * np.log(d.clip(1e-12))
        w = np.exp(log_w-log_w.max(1, keepdims=True))
        u = w / w.sum(1, keepdims=True)
        zeros = d <= 1e-12
        rows = zeros.any(1)
        u[rows] = zeros[rows] / zeros[rows].sum(1, keepdims=True)
        return u
    for _ in range(iterations):
        u = memberships(medoids)
        # Coordinate medoid updates enforce uniqueness without collapsing clusters.
        new = medoids.copy()
        for k in range(clusters):
            costs = (u[:, k] ** fuzziness) @ (distance ** 2)
            costs[np.delete(new, k)] = np.inf
            new[k] = int(costs.argmin())
        if np.array_equal(new, medoids):
            break
        medoids = new
    return memberships(medoids), medoids


class Runner:
    def __init__(self, config, resume=False):
        self.c = config.validate()
        self.out = Path(config.output_root) / config.dataset / f'alpha_{config.alpha:g}' / f'seed_{config.seed}' / f'{config.method}-{config.identity()}'
        self.out.mkdir(parents=True, exist_ok=True)
        self.resume = resume
        self.device = torch.device(config.device)
        if config.device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA requested but unavailable')
        if config.device == 'mps' and not torch.backends.mps.is_available():
            raise RuntimeError('MPS requested but unavailable')
        torch.set_num_threads(config.threads)
        seed_all(config.seed, config.deterministic)
        self.data = prepare(config)
        self.n = len(self.data.clients)
        self.classes = len(self.data.core.CLASS_NAMES)
        self.domains = len(self.data.core.DOMAIN_NAMES)
        if config.clusters > self.n:
            raise ValueError('More clusters than clients')
        self.model = Network(self.classes, config.backbone, config.pretrained).to(self.device)
        initial = cpu_state(self.model)
        self.bn_keys = self.model.bn_keys()
        self.state = dict(round=0, servers=[initial], bank={}, graph=None, bn={}, personal=None,
            membership=np.ones((self.n, 1)), gamma=None, prior=None, bytes_up=0, bytes_down=0,
            history=[], monitor=Monitor(config.monitor_window, config.early_stop_patience, config.min_delta, config.early_stop_min_rounds), best=None,
            cluster_monitors=[], stopped=[], cluster_best=[], elapsed=0., cluster_metrics=None, training_complete=False,
            warmup_monitor=Monitor(config.monitor_window, config.early_stop_patience, config.min_delta, config.early_stop_min_rounds),
            warmup_best=None, warmup_end_round=None, warmup_stop_reason=None)
        if config.method in ('fedrc', 'fedsoft'):
            self.state['membership'] = np.full((self.n, config.clusters), 1/config.clusters)
            self.state['servers'] = [initial]
            for k in range(1, config.clusters):
                # Shared encoder, independently initialized heads break EM symmetry.
                seed_all(config.seed + k * 499, config.deterministic)
                candidate = copy.deepcopy(initial)
                nn.init.kaiming_uniform_(self.model.classifier.weight, a=math.sqrt(5))
                nn.init.uniform_(self.model.classifier.bias, -1/math.sqrt(self.model.feature_dim), 1/math.sqrt(self.model.feature_dim))
                candidate.update({f'classifier.{key}': value.detach().cpu().clone() for key,value in self.model.classifier.state_dict().items()})
                self.state['servers'].append(candidate)
            counts = np.sum([x['counts'] for x in self.data.clients], axis=0).astype(float)
            self.state['prior'] = np.tile((counts+1e-12)/(counts.sum()+self.classes*1e-12), (config.clusters,1))
        checkpoint = self.out / 'last.pt'
        if checkpoint.exists():
            if not resume:
                raise FileExistsError(f'Run exists: {self.out}. Use --resume to continue.')
            saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
            if saved['config'] != asdict(config) or saved['code_hash'] != code_hash():
                raise ValueError('Configuration or source changed; use a new output root instead of unsafe resume')
            if saved['manifest'] != self.data.manifest:
                raise ValueError('Dataset or partition changed since the checkpoint')
            self.state = saved['state']
        atomic_json(self.out / 'config.json', asdict(config))
        atomic_json(self.out / 'partition.json', self.data.manifest)
        atomic_json(self.out / 'environment.json', dict(python=platform.python_version(), torch=torch.__version__,
            numpy=np.__version__, device=str(self.device), code_hash=code_hash(),
            method_status='independent common-protocol implementation; see docs/BASELINES.md'))

    def load(self, state):
        self.model.load_state_dict(state)
        self.model.to(self.device)

    def local(self, initial, cid, round_id, *, graph=None, bank=None, head_only=False,
              weights=None, proximal=None, proximal_mass=1., discovery=False):
        c = self.c
        seed_all(c.seed + 100003 * round_id + 97 * cid, c.deterministic)
        self.load(initial)
        self.model.train()
        self.model.encoder.requires_grad_(not head_only)
        if head_only:
            self.model.encoder.eval()
        groups = [{'params': self.model.classifier.parameters(), 'lr': c.head_lr}]
        if not head_only:
            groups.append({'params': self.model.encoder.parameters(), 'lr': c.encoder_lr})
        optimizer = torch.optim.SGD(groups, momentum=c.momentum, weight_decay=c.weight_decay)
        scaler = torch.amp.GradScaler('cuda', enabled=c.amp)
        class_weights = None
        if c.method.startswith('ours'):
            counts = torch.tensor(self.data.clients[cid]['counts'], dtype=torch.float32, device=self.device)
            class_weights = counts.clamp_min(1).pow(-c.class_gamma)
            class_weights /= class_weights.mean()
        prepared = prepare_bank(bank, c.method, self.device)
        graph = bank_to(graph, self.device)
        target = None if proximal is None else {k: v.to(self.device) for k,v in proximal.items() if k in dict(self.model.named_parameters())}
        total, seen = 0., 0
        loader = self.data.loader(cid, 'train', round_id, two_views=discovery and not head_only)
        for _ in range(c.local_epochs):
            for x, x2, y, domains, positions in loader:
                # Tiny final batches at 32px can leave only one BN value/channel.
                # Preserve the sample by freezing BN statistics for that batch.
                for module in self.model.modules():
                    if isinstance(module, nn.modules.batchnorm._BatchNorm):
                        module.train(not head_only and len(y) > 1)
                x, y, domains = x.to(self.device, non_blocking=True), y.to(self.device), domains.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=self.device.type, enabled=c.amp, dtype=torch.float16):
                    features, logits = self.model(x)
                    losses = F.cross_entropy(logits, y, weight=class_weights, reduction='none')
                    loss = F.cross_entropy(logits, y, weight=class_weights) if weights is None else (losses * weights[positions].to(self.device)).mean()
                    if discovery and not head_only:
                        f2, logits2 = self.model(x2.to(self.device, non_blocking=True))
                        loss = .5 * (loss + F.cross_entropy(logits2, y, weight=class_weights))
                        ramp = min(1., (round_id-c.head_only_rounds)/c.ramp_rounds)
                        loss = loss + ours_loss(features.float(), f2.float(), y, domains, graph, self.data.core, c, ramp)
                    elif prepared is not None:
                        loss = loss + prototype_loss(features.float(), y, self.data.clients[cid]['domain'], prepared, c.method, c)
                    if target is not None:
                        penalty = sum((p.float()-target[name].float()).square().sum() for name,p in self.model.named_parameters())
                        loss = loss + .5*c.soft_proximal*proximal_mass*penalty
                if not torch.isfinite(loss):
                    raise FloatingPointError(f'Nonfinite loss at client {cid}, round {round_id}')
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                total += float(loss.detach()) * len(y)
                seen += len(y)
        self.model.encoder.requires_grad_(True)
        return cpu_state(self.model), total/max(seen, 1)

    def personal_state(self, cid):
        if self.state['personal'] is None:
            return self.state['servers'][0]
        return torch.load(self.out / self.state['personal'] / f'{cid}.pt', map_location='cpu', weights_only=True)

    @torch.inference_mode()
    def predictions(self, state, cid, split, proto=None):
        self.load(state)
        self.model.eval()
        probabilities, targets = [], []
        prepared = prepare_bank(proto, 'fedproto', self.device)
        for x, _, y, _, _ in self.data.loader(cid, split):
            features, logits = self.model(x.to(self.device))
            if prepared is not None:
                values, labels, _ = prepared
                logits = torch.full((len(y), self.classes), -1e9, device=self.device)
                logits[:, labels] = -torch.cdist(features.float(), values.float()).square() / features.shape[1]
            probabilities.append(logits.softmax(1).cpu())
            targets.append(y)
        return (torch.cat(probabilities), torch.cat(targets)) if probabilities else (torch.empty(0, self.classes), torch.empty(0, dtype=torch.long))

    def evaluate(self, split, snapshot=None, cluster=None):
        s = self.state if snapshot is None else snapshot
        confusions = np.zeros((self.domains, self.classes, self.classes), dtype=float)
        captured = {}
        for client in self.data.clients:
            cid, d = client['id'], client['domain']
            if cluster is not None:
                weight = float(s['membership'][cid, cluster])
                if weight < self.c.min_membership:
                    continue
                predictions, y = self.predictions(s['servers'][cluster], cid, split)
            elif self.c.method in ('fedproto', 'fedsoft'):
                local = s['servers'][0] if s['personal'] is None else torch.load(self.out / s['personal'] / f'{cid}.pt', map_location='cpu', weights_only=True)
                predictions, y = self.predictions(local, cid, split, s['bank'] if self.c.method == 'fedproto' else None)
                weight = 1.
            elif self.c.method == 'fedbn':
                local = dict(s['servers'][0])
                local.update(s['bn'].get(cid, {}))
                predictions, y = self.predictions(local, cid, split)
                weight = 1.
            else:
                predictions = None
                y = None
                for k, server in enumerate(s['servers']):
                    u = float(s['membership'][cid, k])
                    if u == 0:
                        continue
                    p, y = self.predictions(server, cid, split)
                    predictions = u*p if predictions is None else predictions+u*p
                weight = 1.
            if split == 'test' and os.environ.get('FEDGAPS_CAPTURE_PREDICTIONS') == '1' and y is not None:
                captured[f'client_{cid}_probabilities'] = predictions.numpy().copy()
                captured[f'client_{cid}_labels'] = y.numpy().copy()
            if y is not None and len(y):
                bins = np.bincount((y.numpy()*self.classes + predictions.argmax(1).numpy()), minlength=self.classes**2)
                confusions[d] += weight * bins.reshape(self.classes, self.classes)
        if captured:
            np.savez_compressed(self.out / 'test_predictions.npz', **captured)
        return predictive(confusions, self.data.core.DOMAIN_NAMES)

    def snapshot(self):
        return copy.deepcopy({key:self.state[key] for key in ('servers','bank','bn','personal','membership')})

    def standard_round(self, r):
        c, s = self.c, self.state
        average = Average(self.bn_keys if c.method == 'fedbn' else ())
        local_protos, losses = {}, []
        personal = f'clients/round_{r}' if c.method == 'fedproto' else None
        initial = s['servers'][0]
        for client in self.data.clients:
            cid = client['id']
            source = self.personal_state(cid) if c.method == 'fedproto' else dict(initial)
            if c.method == 'fedbn':
                source.update(s['bn'].get(cid, {}))
            trained, loss = self.local(source, cid, r, bank=s['bank'])
            losses.append(loss)
            if c.method != 'fedproto':
                shared = {k:v for k,v in trained.items() if c.method != 'fedbn' or k not in self.bn_keys}
                s['bytes_up'] += tensor_bytes(shared)
                s['bytes_down'] += tensor_bytes(shared)
                average.add(trained, len(client['train']))
            else:
                atomic_torch(self.out / personal / f'{cid}.pt', trained)
            if c.method == 'fedbn':
                s['bn'][cid] = {k:v for k,v in trained.items() if k in self.bn_keys}
            if c.method in ('fedproto','fedplvm','fpl','feddap'):
                local_protos[cid] = collect(self.model, self.data.loader(cid,'prototype'), self.device, multi=c.method=='fedplvm')
                s['bytes_up'] += tensor_bytes(local_protos[cid]) + (16 if c.method=='feddap' else 8)*sum(len(v) for v in local_protos[cid].values())
                s['bytes_down'] += tensor_bytes(s['bank']) + (16 if c.method=='feddap' else 8)*sum(len(v) for v in s['bank'].values())
        if c.method != 'fedproto':
            s['servers'] = [average.result(initial)]
        if local_protos:
            s['bank'] = aggregate(local_protos, c.method, [x['domain'] for x in self.data.clients], c)
        s['personal'] = personal
        return float(np.mean(losses))

    def discovery_round(self, r):
        c, s = self.c, self.state
        average = Average()
        losses = []
        initial = s['servers'][0]
        domain_totals = np.bincount([x['domain'] for x in self.data.clients], weights=[len(x['train']) for x in self.data.clients])
        for client in self.data.clients:
            trained, loss = self.local(initial, client['id'], r, graph=s['graph'],
                head_only=r<=c.head_only_rounds, discovery=True)
            average.add(trained, len(client['train']) / domain_totals[client['domain']])
            losses.append(loss)
            s['bytes_up'] += tensor_bytes(trained)
            s['bytes_down'] += tensor_bytes(initial)
            if s['graph'] is not None:
                s['bytes_down'] += tensor_bytes(vars(s['graph']))
        s['servers'] = [average.result(initial)]
        self.load(s['servers'][0])
        stats = signatures(self.model, self.data, c, self.device)
        # Signatures use the new shared coordinate system: a second broadcast is required.
        s['bytes_down'] += self.n*tensor_bytes(s['servers'][0])
        s['bytes_up'] += sum(len(x)*(self.model.feature_dim*4+8+4) for x in stats)
        s['graph'] = graph_bank(stats, self.data, self.model, c)
        from .diagnostics import export_stats
        export_stats(self, stats, s['graph'], f'round_{r:03d}', r)
        return float(np.mean(losses))

    def discover(self, stats=None):
        c, s = self.c, self.state
        self.load(s['servers'][0])
        if stats is None:
            stats = signatures(self.model, self.data, c, self.device)
            s['bytes_down'] += self.n*tensor_bytes(s['servers'][0])
            s['bytes_up'] += sum(len(x)*(self.model.feature_dim*4+12) for x in stats)
        from .diagnostics import export_stats
        export_stats(self, stats, None, 'selected', s['warmup_monitor'].best_round)
        distance = self.data.core.proposed_distance(stats, c.prototype_mix, c.overlap_penalty)
        if c.method == 'ours_hard':
            hard = self.data.core.hard_cluster(distance, c.clusters)
            membership = np.eye(c.clusters)[hard]
        else:
            membership, _ = fuzzy_membership(distance, c.clusters, c.fuzziness, self.data.core)
        # Use exactly the active memberships both for training and inference.
        active = membership >= c.min_membership
        active[np.arange(self.n), membership.argmax(1)] = True
        membership = membership*active
        membership /= membership.sum(1, keepdims=True)
        s['membership'] = membership
        s['servers'] = [copy.deepcopy(s['servers'][0]) for _ in range(c.clusters)]
        s['cluster_best'] = copy.deepcopy(s['servers'])
        s['cluster_monitors'] = [Monitor(c.monitor_window,c.early_stop_patience,c.min_delta,c.early_stop_min_rounds) for _ in range(c.clusters)]
        s['stopped'] = [False]*c.clusters
        s['cluster_metrics'] = clustering(membership,self.data.clients,distance)
        s['cluster_metrics']['silhouette_space'] = 'proposed prototype-support distance'
        np.save(self.out / 'distance.npy',distance)
        np.save(self.out / 'membership.npy',membership)
        atomic_json(self.out / 'clustering.json',s['cluster_metrics'])

    def cluster_round(self, r):
        c, s = self.c, self.state
        losses = []
        for k, initial in enumerate(s['servers']):
            if s['stopped'][k]:
                continue
            if not np.any(s['membership'][:,k] > 0):
                s['stopped'][k] = True
                continue
            average = Average()
            for client in self.data.clients:
                cid = client['id']
                weight = s['membership'][cid,k]
                if weight == 0:
                    continue
                trained, loss = self.local(initial,cid,r)
                average.add(trained,len(client['train'])*weight)
                s['bytes_up'] += tensor_bytes(trained)
                s['bytes_down'] += tensor_bytes(initial)
                losses.append(loss)
            s['servers'][k] = average.result(initial)
            metrics = self.evaluate('validation',cluster=k)
            score = .5*(metrics['balanced_accuracy']+metrics['worst_class_recall'])
            improved, stop = s['cluster_monitors'][k].update(score,r-s['warmup_end_round'])
            if improved:
                s['cluster_best'][k] = copy.deepcopy(s['servers'][k])
            if stop and c.method != 'ours_fixed_rounds':
                s['stopped'][k] = True
                s['servers'][k] = copy.deepcopy(s['cluster_best'][k])
        return float(np.mean(losses)) if losses else 0.

    def mixture_round(self, r):
        c, s = self.c, self.state
        K = c.clusters
        sizes = np.array([len(x['train']) for x in self.data.clients])
        gamma, new_prior = [], np.zeros((K,self.classes))
        reestimate = c.method == 'fedrc' or (r-1)%c.soft_update_interval == 0
        if reestimate:
            for client in self.data.clients:
                cid = client['id']
                log_likelihood = []
                for server in s['servers']:
                    p, y = self.predictions(server,cid,'prototype')
                    log_likelihood.append(p[torch.arange(len(y)), y].clamp_min(1e-30).log())
                ll = torch.stack(log_likelihood,1).numpy()
                if c.method == 'fedrc':
                    g = rc_responsibilities(ll,s['membership'][cid],s['prior'],y.numpy())
                    gamma.append(torch.tensor(g,dtype=torch.float32))
                    s['membership'][cid] = g.mean(0)
                    for k in range(K):
                        np.add.at(new_prior[k],y.numpy(),g[:,k])
                else:
                    counts = np.bincount(ll.argmax(1),minlength=K)
                    # Preserve paper's unnormalized max(counter/n, sigma) weights.
                    s['membership'][cid] = np.maximum(counts/len(y),c.soft_counter_floor)
                s['bytes_down'] += sum(tensor_bytes(x) for x in s['servers'])
                if c.method == 'fedrc':
                    s['bytes_down'] += K*self.classes*8
                s['bytes_up'] += K*8 + (K*self.classes*8 if c.method=='fedrc' else 0)
            if c.method == 'fedrc':
                s['prior'] = (new_prior+1e-12)/(new_prior.sum(1,keepdims=True)+self.classes*1e-12)
        losses = []
        averages = [Average() for _ in range(K)]
        personal = f'clients/round_{r}' if c.method=='fedsoft' else None
        for client in self.data.clients:
            cid = client['id']
            if c.method == 'fedsoft':
                target_avg = Average()
                for k,server in enumerate(s['servers']):
                    target_avg.add(server,float(s['membership'][cid,k]))
                target = target_avg.result(s['servers'][0])
                trained,loss = self.local(self.personal_state(cid),cid,r,proximal=target,
                    proximal_mass=float(s['membership'][cid].sum()))
                atomic_torch(self.out / personal / f'{cid}.pt',trained)
                for k in range(K):
                    averages[k].add(trained,sizes[cid]*s['membership'][cid,k])
                s['bytes_up'] += tensor_bytes(trained)
                losses.append(loss)
                if not reestimate:
                    s['bytes_down'] += sum(tensor_bytes(x) for x in s['servers'])
            else:
                for k,server in enumerate(s['servers']):
                    trained,loss = self.local(server,cid,r,weights=gamma[cid][:,k])
                    # Eq. 5 already weights gradients by gamma. Eq. 6 uses N_i, not N_i*omega.
                    averages[k].add(trained,sizes[cid])
                    s['bytes_up'] += tensor_bytes(trained)
                    losses.append(loss)
        s['servers'] = [avg.result(old) for avg,old in zip(averages,s['servers'])]
        s['personal'] = personal
        return float(np.mean(losses))

    def checkpoint(self):
        atomic_torch(self.out / 'last.pt',dict(config=asdict(self.c),code_hash=code_hash(),
            manifest=self.data.manifest,state=self.state))
        # Immutable generations make interrupted personalized rounds safe to replay.
        keep = {self.state['personal']}
        if self.state['best'] is not None:
            keep.add(self.state['best']['personal'])
        for path in (self.out / 'clients').glob('round_*') if (self.out / 'clients').exists() else []:
            if str(path.relative_to(self.out)) not in keep:
                shutil.rmtree(path)

    def finish_warmup(self, round_id, reason):
        """Restore selected discovery weights before deriving cluster memberships."""
        s = self.state
        if s['warmup_best'] is not None:
            s['servers'] = [copy.deepcopy(s['warmup_best'])]
        s['warmup_end_round'] = round_id
        s['warmup_stop_reason'] = reason
        self.discover()  # Re-extract signatures from the restored encoder, not the last round.
        s['warmup_best'] = None
        print(f"Warm-up ended at round {round_id}; selected round "
              f"{s['warmup_monitor'].best_round}; reason={reason}", flush=True)

    def run(self):
        c, s = self.c,self.state
        if (self.out / 'metrics.json').exists() and self.resume:
            return json.loads((self.out / 'metrics.json').read_text())
        ours = c.method.startswith('ours')
        if ours and c.warmup_rounds == 0 and s['warmup_end_round'] is None:
            self.finish_warmup(0, 'disabled')
        from .diagnostics import export_initial
        if ours and s['round'] == 0:
            export_initial(self)
        start = time.monotonic()
        previous_elapsed = s['elapsed']
        for r in range(s['round']+1,c.rounds+1):
            if s['training_complete'] or (s['stopped'] and all(s['stopped'])):
                break
            warmup = ours and s['warmup_end_round'] is None
            phase = 'warmup' if warmup else 'cluster' if ours else 'baseline'
            if warmup:
                loss = self.discovery_round(r)
            elif ours:
                loss = self.cluster_round(r)
            elif c.method in ('fedrc','fedsoft'):
                loss = self.mixture_round(r)
            else:
                loss = self.standard_round(r)
            validation = self.evaluate('validation')
            score = .5*(validation['balanced_accuracy']+validation['worst_class_recall'])
            if warmup:
                improved, stop = s['warmup_monitor'].update(score,r)
                if improved:
                    s['warmup_best'] = copy.deepcopy(s['servers'][0])
                if stop or r == c.warmup_rounds:
                    self.finish_warmup(r, 'patience' if stop else 'round_cap')
            elif not ours:
                improved, stop = s['monitor'].update(score,r)
                if improved:
                    s['best'] = self.snapshot()
                s['training_complete'] = stop or r == c.rounds
            else:
                # Stopping warm-up sooner does not enlarge the cluster-stage budget.
                cluster_cap = c.rounds - c.warmup_rounds
                s['training_complete'] = all(s['stopped']) or r-s['warmup_end_round'] >= cluster_cap
            s['round'] = r
            s['elapsed'] = previous_elapsed + time.monotonic()-start
            s['history'].append(dict(round=r,phase=phase,loss=loss,validation_score=score,
                accuracy=validation['accuracy'],bytes_up=s['bytes_up'],bytes_down=s['bytes_down']))
            atomic_json(self.out / 'history.json',s['history'])
            print(f'{c.method} {phase} round {r}/{c.rounds} loss={loss:.4f} validation={score:.4f}',flush=True)
            # Terminal checkpoints include stop decisions and completed phase transitions.
            if r%c.checkpoint_every==0 or s['personal'] is not None or s['training_complete'] or (warmup and s['warmup_end_round'] is not None):
                self.checkpoint()
            if s['training_complete']:
                break
        if c.method.startswith('ours'):
            final = self.snapshot()
            final['servers'] = copy.deepcopy(s['cluster_best'])
            selected_rounds = [m.export() for m in s['cluster_monitors']]
        else:
            final = s['best']
            selected_rounds = [s['monitor'].export()]
        final_metrics = self.evaluate('test',final)
        cluster_metrics = s['cluster_metrics']
        if c.method in ('fedrc','fedsoft'):
            u = final['membership']
            u = u/u.sum(1,keepdims=True)
            distance = np.sqrt(np.maximum(((u[:,None]-u[None])**2).sum(2),0))
            cluster_metrics = clustering(u,self.data.clients,distance)
            cluster_metrics['silhouette_space'] = 'client membership Euclidean distance'
            np.save(self.out / 'membership.npy',u)
        result = dict(config=asdict(c), predictive=final_metrics, clustering=cluster_metrics,
            selection=selected_rounds, rounds_executed=s['round'], seconds=s['elapsed'],
            warmup_selection=({**s['warmup_monitor'].export(), 'stop_reason':s['warmup_stop_reason'],
                'stopped_round':s['warmup_end_round']} if ours else None),
            stopping_policy=dict(patience=c.early_stop_patience,min_delta=c.min_delta,
                window=c.monitor_window,minimum_rounds=c.early_stop_min_rounds,
                fixed_cluster_rounds=c.method=='ours_fixed_rounds'),
            communication=dict(upload_bytes=s['bytes_up'],download_bytes=s['bytes_down'],
                total_bytes=s['bytes_up']+s['bytes_down'], accounting='uncompressed training tensor payloads; excludes validation/test and transport overhead'),
            partition=str(self.data.manifest_path), status='completed',synthetic=c.dataset=='synthetic',
            implementation='independent common-protocol reproduction, not official benchmark scores')
        atomic_torch(self.out / 'selected.pt',dict(config=asdict(c),snapshot=final))
        atomic_json(self.out / 'metrics.json',result)
        return result


def run(config,resume=False):
    output = Path(config.output_root) / config.dataset / f'alpha_{config.alpha:g}' / f'seed_{config.seed}' / f'{config.method}-{config.identity()}'
    output.mkdir(parents=True,exist_ok=True)
    # OS lock is released after crashes and prevents concurrent writers to one run.
    import fcntl
    with (output / '.lock').open('w') as lock:
        try:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f'Run already active: {output}') from error
        try:
            return Runner(config,resume).run()
        except Exception as error:
            atomic_json(output / 'failure.json',dict(type=type(error).__name__,message=str(error)))
            raise
