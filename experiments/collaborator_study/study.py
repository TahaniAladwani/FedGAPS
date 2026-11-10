import copy
import csv
import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
from icde.engine import Runner
from icde.prototypes import signatures
from icde.metrics import Monitor
from icde.utils import atomic_json, atomic_torch, code_hash, tensor_bytes
from comparisons import packed_stats, summary_bytes, distance_for, discovery_metrics
from grouping import assign, state_hash


def add(ledger, name, up=0, down=0):
    item=ledger.setdefault(name,dict(upload_bytes=0,download_bytes=0))
    item['upload_bytes']+=int(up); item['download_bytes']+=int(down)


def totals(ledger):
    up=sum(v['upload_bytes'] for v in ledger.values())
    down=sum(v['download_bytes'] for v in ledger.values())
    return dict(upload_bytes=up,download_bytes=down,total_bytes=up+down)


def merge_ledger(target, source, prefix=''):
    for name,values in source.items():
        add(target,prefix+name,values['upload_bytes'],values['download_bytes'])


def hash_manifest(manifest):
    return hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()


def write_csv(path, rows):
    if not rows: return
    with Path(path).open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


class AccountingRunner(Runner):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.state.setdefault('communication_components',{})

    def sync_bytes(self):
        value=totals(self.state['communication_components'])
        self.state['bytes_up']=value['upload_bytes']; self.state['bytes_down']=value['download_bytes']

    def discovery_round(self,r):
        model_size=tensor_bytes(self.state['servers'][0])
        graph_bytes=0 if self.state['graph'] is None else self.n*tensor_bytes(vars(self.state['graph']))
        loss=super().discovery_round(r)
        ledger=self.state['communication_components']
        add(ledger,'discovery_models',up=self.n*model_size,down=2*self.n*model_size)
        add(ledger,'discovery_summaries',up=summary_bytes(self.n,self.classes,self.model.feature_dim,'full'),down=graph_bytes)
        self.sync_bytes()
        return loss

    def cluster_round(self,r):
        up,down=self.state['bytes_up'],self.state['bytes_down']
        loss=super().cluster_round(r)
        add(self.state['communication_components'],'downstream_models',
            self.state['bytes_up']-up,self.state['bytes_down']-down)
        self.sync_bytes()
        return loss

    def standard_round(self,r):
        up,down=self.state['bytes_up'],self.state['bytes_down']
        loss=super().standard_round(r)
        add(self.state['communication_components'],'downstream_models',
            self.state['bytes_up']-up,self.state['bytes_down']-down)
        self.sync_bytes()
        return loss

    def run(self):
        result=super().run()
        ledger=self.state['communication_components']
        result['communication']={**totals(ledger), 'components':ledger,
            'accounting':'uncompressed model tensor payloads plus explicit dense summary schema; excludes validation/test, transport and control messages; not measured network traffic'}
        result['partition_hash']=hash_manifest(self.data.manifest)
        if (self.out/'comparison.json').exists():
            result['comparison']=json.loads((self.out/'comparison.json').read_text())
        atomic_json(self.out/'metrics.json',result)
        return result


class DiscoveryComplete(Exception): pass


class DiscoveryRunner(AccountingRunner):
    def finish_warmup(self,round_id,reason):
        s=self.state
        selected=s['warmup_best'] if s['warmup_best'] is not None else s['servers'][0]
        self.load(selected)
        stats=signatures(self.model,self.data,self.c,self.device)
        pack=packed_stats(stats,self.classes,self.model.feature_dim)
        started=time.perf_counter()
        distance=self.data.core.proposed_distance(stats,self.c.prototype_mix,self.c.overlap_penalty)
        distance_seconds=time.perf_counter()-started
        cache=dict(config=asdict(self.c),source_hash=code_hash(),manifest=self.data.manifest,
                   model=selected,initialization_hash=state_hash(selected),
                   encoder_hash=state_hash({k:v for k,v in selected.items() if k.startswith('encoder.')}),
                   distance=distance,distance_seconds=distance_seconds,
                   vectors=self.data.core.signature_vectors(stats),**pack,
                   stopped_round=round_id,selected_round=s['warmup_monitor'].best_round,
                   warmup_monitor=s['warmup_monitor'],feature_dim=self.model.feature_dim,
                   seconds=self.previous_seconds+time.monotonic()-self.started,
                   communication_components=copy.deepcopy(s['communication_components']))
        atomic_torch(self.cache_path,cache)
        atomic_json(self.cache_path.with_suffix('.json'),{k:cache[k] for k in
            ('config','source_hash','initialization_hash','encoder_hash','stopped_round','selected_round',
             'seconds','distance_seconds','communication_components')})
        raise DiscoveryComplete()

    def obtain(self,path):
        self.cache_path=Path(path)
        self.started=time.monotonic(); self.previous_seconds=self.state['elapsed']
        if self.cache_path.exists():
            cache=torch.load(self.cache_path,map_location='cpu',weights_only=False)
            if cache['config']!=asdict(self.c) or cache['manifest']!=self.data.manifest or cache['source_hash']!=code_hash():
                raise ValueError('Incompatible discovery cache. Use a new output directory.')
            return cache
        try: self.run()
        except DiscoveryComplete: pass
        return torch.load(self.cache_path,map_location='cpu',weights_only=False)


class MatchedRunner(AccountingRunner):
    def start_from(self,aligned,unaligned=None):
        c,s=self.c,self.state
        if self.data.manifest!=aligned['manifest'] or (unaligned is not None and unaligned['manifest']!=aligned['manifest']):
            raise ValueError('Client partitions differ')
        expected_source=unaligned if c.comparison=='unaligned' else aligned
        if s['round']>0:
            previous=json.loads((self.out/'comparison.json').read_text())
            if previous['downstream_initialization_hash']!=aligned['initialization_hash'] or previous['signature_encoder_hash']!=expected_source['encoder_hash']:
                raise ValueError('Cached downstream run uses a different discovery checkpoint')
            return
        started=time.perf_counter()
        distance=distance_for(c.comparison,aligned,unaligned,self.data.core)
        distance_seconds=time.perf_counter()-started
        if c.comparison in ('aligned','unaligned'):
            distance_seconds=expected_source['distance_seconds']
        started=time.perf_counter()
        u=assign(distance,expected_source['vectors'],c.grouping,c.clusters,c.seed,
                 self.data.core,c.fuzziness,c.min_membership)
        clustering_seconds=time.perf_counter()-started
        s['servers']=[copy.deepcopy(aligned['model']) for _ in range(c.clusters)]
        s['membership']=u
        s['round']=aligned['stopped_round']; s['warmup_end_round']=aligned['stopped_round']
        s['warmup_stop_reason']='matched_shared_checkpoint'
        s['warmup_monitor']=copy.deepcopy(aligned['warmup_monitor'])
        s['cluster_best']=copy.deepcopy(s['servers'])
        s['cluster_monitors']=[Monitor(c.monitor_window,c.early_stop_patience,c.min_delta,c.early_stop_min_rounds) for _ in range(c.clusters)]
        s['stopped']=[False]*c.clusters
        s['cluster_metrics'],neighbors=discovery_metrics(u,distance,self.data.clients,c.neighbor_k)
        ledger={}
        merge_ledger(ledger,aligned['communication_components'])
        s['elapsed']=aligned['seconds']
        if c.comparison=='unaligned':
            # The study uses aligned initialization AND an auxiliary unaligned encoder.
            merge_ledger(ledger,unaligned['communication_components'],prefix='auxiliary_')
            s['elapsed']+=unaligned['seconds']
        kind='counts' if c.comparison=='counts' else 'support' if c.comparison=='reliability_counts' else 'full'
        needs_encoder=kind=='full' or (kind=='support' and c.tau_variance>0)
        add(ledger,'selection_models',down=self.n*tensor_bytes(expected_source['model']) if needs_encoder else 0)
        # Counts were already uploaded during common discovery. With tau_variance=0,
        # reliability-tempered support is also computable from these cached counts.
        selection_up=summary_bytes(self.n,self.classes,aligned['feature_dim'],kind) if needs_encoder else 0
        add(ledger,'selection_summaries',up=selection_up,down=u.nbytes)
        s['communication_components']=ledger
        self.sync_bytes()
        info=dict(variant=c.comparison,grouping=c.grouping,k=c.clusters,
                  downstream_initialization_hash=aligned['initialization_hash'],
                  signature_encoder_hash=expected_source['encoder_hash'],
                  common_selected_round=aligned['selected_round'],signature_selected_round=expected_source['selected_round'],
                  partition_hash=hash_manifest(self.data.manifest),
                  distance_seconds=distance_seconds,clustering_seconds=clustering_seconds,
                  histogram_metric='Hellinger distance',neighbor_k=c.neighbor_k,
                  shared_discovery_charged=True,auxiliary_discovery_charged=c.comparison=='unaligned')
        atomic_json(self.out/'comparison.json',info)
        atomic_json(self.out/'clustering.json',s['cluster_metrics'])
        np.save(self.out/'distance.npy',distance); np.save(self.out/'membership.npy',u)
        write_csv(self.out/'neighbors.csv',neighbors)
        self.checkpoint()


def audit_partition(data,output,seed):
    rows=[]; classes=len(data.core.CLASS_NAMES)
    for c in data.clients:
        if set(c['train'])&set(c['validation']): raise ValueError('Train/validation overlap')
        rows.append(dict(client=c['id'],domain=data.core.DOMAIN_NAMES[c['domain']],
                         planted_group=c.get('planted_group',-1),train_samples=len(c['train']),
                         first_half_fraction=sum(c['counts'][:classes//2])/sum(c['counts'])))
    for d in range(len(data.core.DOMAIN_NAMES)):
        local=[c for c in data.clients if c['domain']==d]
        alltrain=[i for c in local for i in c['train']+c['validation']]
        if len(alltrain)!=len(set(alltrain)): raise ValueError('Duplicated train/validation samples')
        ids=[i for c in local for i in c['test']]
        if len(ids)!=len(set(ids)): raise ValueError('Duplicated test samples')
    write_csv(Path(output)/f'client_manifest_seed{seed}.csv',rows)
    domains=[sum(len(c['train']) for c in data.clients if c['domain']==d) for d in range(len(data.core.DOMAIN_NAMES))]
    atomic_json(Path(output)/f'partition_audit_seed{seed}.json',dict(
        partition_hash=hash_manifest(data.manifest),domain_training_totals=domains,
        domain_balanced_control_distinct=len(set(domains))>1,
        label_definition=data.manifest['semantic_label_definition'],split_disjointness_verified=True))
