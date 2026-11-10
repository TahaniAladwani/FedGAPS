"""Check completed reports without changing training artifacts."""
import argparse
import json
from pathlib import Path
import numpy as np
from comparisons import hellinger


def verify(root):
    root=Path(root)
    plan=json.loads((root/'plan.json').read_text())
    paths=list(root.glob('runs/**/metrics.json'))
    expected=sum(j['stage']!='discovery' for j in plan['jobs'])
    assert len(paths)==expected,(len(paths),expected)
    for path in paths:
        result=json.loads(path.read_text());c=result['config'];comm=result['communication']
        for direction in ('upload','download'):
            assert sum(v[direction+'_bytes'] for v in comm['components'].values())==comm[direction+'_bytes']
        assert comm['total_bytes']==comm['upload_bytes']+comm['download_bytes']
        cache=json.loads((root/'cache'/c['partition_mode']/f"aligned_seed{c['seed']}.json").read_text())
        partition=json.loads((path.parent/'partition.json').read_text())
        n=len(partition['clients'])
        b=cache['communication_components']['discovery_models']['upload_bytes']//(n*cache['stopped_round'])
        if c['method']=='fedavg':
            updates=n*c['rounds']
        else:
            info=result['comparison']
            assert info['downstream_initialization_hash']==cache['initialization_hash']
            if c['comparison']!='unaligned':assert info['signature_encoder_hash']==cache['encoder_hash']
            u=np.load(path.parent/'membership.npy');d=np.load(path.parent/'distance.npy')
            np.testing.assert_allclose(u.sum(1),1.)
            np.testing.assert_allclose(d,d.T,atol=1e-10)
            assert np.isfinite(u).all() and np.isfinite(d).all()
            updates=int((u>0).sum())*(c['rounds']-c['warmup_rounds'])
            if c['comparison']=='counts':
                expected_d=hellinger([x['counts'] for x in partition['clients']])
                np.testing.assert_allclose(d,expected_d,atol=1e-12)
            if c['partition_mode']=='dirichlet':
                assert result['clustering']['planted_ari'] is None
                assert result['clustering']['cross_domain_precision_at_k'] is None
            else:
                score=result['clustering']['cross_domain_precision_at_k']
                assert 0<=score<=1
        downstream=comm['components']['downstream_models']
        assert downstream['upload_bytes']==updates*b
        assert downstream['download_bytes']==updates*b
    print(f'Verified {len(paths)} completed results: matched initialization, histogram distances, memberships and all cluster-model payloads.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path)
    verify(p.parse_args().output)
