"""Strict, read-only import of the previously delivered planted suite."""
import copy
import json
import time
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
from icde.prototypes import signatures
from icde.utils import atomic_json, atomic_torch, code_hash, tensor_bytes
from comparisons import packed_stats
from grouping import state_hash
from study import add, totals, hash_manifest

NEW_FIELDS={'partition_mode','planted_sampling','comparison','neighbor_k'}


def accepted_hash():
    return json.loads((Path(__file__).parent/'legacy_protocol.json').read_text())['accepted_source_hash']


def same_config(old,current):
    expected={k:v for k,v in asdict(current).items() if k not in NEW_FIELDS|{'output_root'}}
    actual={k:v for k,v in old.items() if k!='output_root'}
    if actual!=expected:
        changed={k:(actual.get(k),expected.get(k)) for k in set(actual)|set(expected) if actual.get(k)!=expected.get(k)}
        raise ValueError(f'Legacy protocol mismatch: {changed}. Match the old settings exactly or use fresh outputs.')


def legacy_cache(root,seed):
    cache=torch.load(Path(root)/f'discovery_seed{seed}.pt',map_location='cpu',weights_only=False)
    if cache['source_hash']!=accepted_hash():
        raise ValueError('Legacy source version is not the verified previous package; automatic reuse rejected.')
    return cache


def import_discovery(runner,root,path):
    c=runner.c; old=legacy_cache(root,c.seed)
    same_config(old['config'],c)
    if old['manifest']!=runner.data.manifest:raise ValueError('Legacy partition/data fingerprint mismatch')
    runner.load(old['model'])
    stats=signatures(runner.model,runner.data,c,runner.device)
    pack=packed_stats(stats,runner.classes,runner.model.feature_dim)
    start=time.perf_counter()
    check=runner.data.core.proposed_distance(stats,c.prototype_mix,c.overlap_penalty)
    elapsed=time.perf_counter()-start
    if not np.allclose(check,old['distance'],atol=1e-6,rtol=1e-5):
        raise ValueError('Re-extracted signatures do not reproduce the legacy distance matrix')
    b=tensor_bytes(old['model']);n=runner.n;r=old['stopped_round']
    old_selection_up=sum(len(x)*(runner.model.feature_dim*4+12) for x in stats)
    summary_up=old['bytes_up']-r*n*b-old_selection_up
    summary_down=old['bytes_down']-2*r*n*b-n*b
    if min(summary_up,summary_down)<0:raise ValueError('Legacy communication counters inconsistent')
    ledger={}
    add(ledger,'discovery_models',up=r*n*b,down=2*r*n*b)
    add(ledger,'discovery_summaries',up=summary_up,down=summary_down)
    cache=dict(config=asdict(c),source_hash=code_hash(),manifest=runner.data.manifest,
               model=old['model'],initialization_hash=state_hash(old['model']),encoder_hash=old['encoder_hash'],
               distance=old['distance'],distance_seconds=elapsed,vectors=old['vectors'],**pack,
               stopped_round=r,selected_round=old['selected_round'],warmup_monitor=old['warmup_monitor'],
               feature_dim=runner.model.feature_dim,seconds=old['seconds'],communication_components=ledger,
               reused_from=str(Path(root).resolve()),
               communication_provenance='Legacy recorded total counters decomposed using fixed all-client discovery model transfers; selected-checkpoint exchange removed and costed per comparison. Legacy summary schema retained.',
               distance_timing_provenance='recomputed diagnostic timing, not original training-run timing')
    atomic_torch(path,cache)
    atomic_json(Path(path).with_suffix('.json'),{k:cache[k] for k in
        ('config','source_hash','initialization_hash','encoder_hash','stopped_round','selected_round','seconds',
         'distance_seconds','communication_components','reused_from','communication_provenance','distance_timing_provenance')})
    print(f'Imported verified planted discovery: seed {c.seed}',flush=True)


def import_result(runner,root,cache):
    c=runner.c
    label=('domain_balanced_fedavg' if c.domain_balanced else 'fedavg') if c.method=='fedavg' else f'{c.grouping}_k{c.clusters}'
    paths=list((Path(root)/label/c.dataset/'planted_70_30'/f'seed_{c.seed}').glob('*/metrics.json'))
    if len(paths)!=1:raise ValueError(f'Need one complete legacy result for {label}, seed {c.seed}; found {len(paths)}')
    folder=paths[0].parent
    old=json.loads(paths[0].read_text());same_config(old['config'],c)
    if json.loads((folder/'partition.json').read_text())!=runner.data.manifest:raise ValueError('Legacy result partition mismatch')
    saved=torch.load(folder/'last.pt',map_location='cpu',weights_only=False)
    if saved['code_hash']!=accepted_hash():raise ValueError('Legacy training source version mismatch')
    selected=torch.load(folder/'selected.pt',map_location='cpu',weights_only=False)
    current_info=json.loads((runner.out/'comparison.json').read_text()) if c.method!='fedavg' else None
    current_metrics=copy.deepcopy(runner.state['cluster_metrics'])
    if c.method!='fedavg':
        encoder=json.loads((folder/'shared_encoder.json').read_text())['encoder_hash']
        if encoder!=cache['encoder_hash']:raise ValueError('Legacy grouped run did not use the shared discovery encoder')
        if not np.allclose(np.load(folder/'membership.npy'),runner.state['membership'],atol=1e-12,rtol=1e-12):
            raise ValueError('New grouping differs from legacy membership; cannot reuse downstream accuracy')
        original_cache=legacy_cache(root,c.seed)
        up=old['communication']['upload_bytes']-original_cache['bytes_up']
        down=old['communication']['download_bytes']-original_cache['bytes_down']
        ledger=copy.deepcopy(runner.state['communication_components'])
    else:
        up=old['communication']['upload_bytes'];down=old['communication']['download_bytes'];ledger={}
    if min(up,down)<0:raise ValueError('Negative inferred downstream communication')
    add(ledger,'downstream_models',up=up,down=down)
    runner.state=saved['state']
    runner.state['communication_components']=ledger
    if c.method!='fedavg':runner.state['cluster_metrics']=current_metrics
    runner.sync_bytes();runner.checkpoint()
    result=copy.deepcopy(old)
    result.update(config=asdict(c),partition=str(runner.data.manifest_path),partition_hash=hash_manifest(runner.data.manifest),
                  clustering=runner.state['cluster_metrics'],reused_from=str(folder.resolve()))
    result['communication']={**totals(ledger),'components':ledger,
         'accounting':'Legacy training payload counters reused; common discovery decomposed by fixed protocol; selection exchange recalculated under the explicit matched schema. Excludes validation/test and transport.'}
    if current_info is not None:result['comparison']=current_info
    selected['config']=asdict(c);atomic_torch(runner.out/'selected.pt',selected)
    atomic_json(runner.out/'metrics.json',result)
    print(f'Imported verified downstream result: {label}, seed {c.seed}',flush=True)
