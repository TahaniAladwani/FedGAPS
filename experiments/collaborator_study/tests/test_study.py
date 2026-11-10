import unittest
import numpy as np
from comparisons import hellinger,support_js,neighbor_recovery,discovery_metrics,summary_bytes
from grouping import assign
from icde.data import reference_core
from icde.planted import allocate_finite,domain_balanced_weight
from study import add,totals,merge_ledger


class StudyTests(unittest.TestCase):
    def test_small_domain_fallback(self):
        capacities=[5,9,12,15,8,10,14,11,7,13]
        y=np.repeat(np.arange(10),capacities)
        for seed in (42,123):
            shards, groups, info=allocate_finite(y,10,4,500,2,4,seed,True)
            self.assertTrue(info['small_domain_fallback'])
            self.assertEqual(info['samples_per_client'],10)
            self.assertTrue(all(len(x)==10 for x in shards))
            ids=sum(shards,[])
            self.assertEqual(len(ids),len(set(ids)))
            self.assertTrue(np.all(np.bincount(y[ids],minlength=10)<=capacities))
            self.assertEqual(sum(groups==0),2)
            self.assertGreaterEqual(info['rejected_draws'],0)

    def test_missing_class_is_not_silently_bypassed(self):
        y=np.repeat(np.arange(9),30)
        with self.assertRaisesRegex(ValueError,'Class counts'):
            allocate_finite(y,10,4,500,2,4,42)

    def test_hellinger_known_values(self):
        d=hellinger([[1,0],[0,1],[5,0],[.5,.5]])
        self.assertAlmostEqual(d[0,1],1.)
        self.assertEqual(d[0,2],0.)
        self.assertAlmostEqual(d[0,3],np.sqrt(1-1/np.sqrt(2)))
        np.testing.assert_array_equal(np.diag(d),0)
        np.testing.assert_allclose(d,d.T)

    def test_exact_existing_support_term(self):
        core=reference_core();p=np.array([[.7,.3],[.2,.8],[.5,.5]])
        d=support_js(p,core)
        self.assertAlmostEqual(d[0,1],core.js_divergence(p[0],p[1]))

    def test_neighbors_exclude_domain_and_do_not_use_observed_histograms(self):
        clients=[dict(id=i,domain=i//2,planted_group=i%2,counts=[9,1]) for i in range(8)]
        d=np.array([[0 if a['planted_group']==b['planted_group'] else 1 for b in clients] for a in clients],float)
        score,rows=neighbor_recovery(d,clients,3)
        self.assertEqual(score,1.)
        self.assertTrue(all(x['eligible_clients']==6 for x in rows))
        # All observed histograms are identical, but labels still come from the generating groups.
        score,_=neighbor_recovery(np.zeros((8,8)),clients,3)
        self.assertAlmostEqual(score,.5)
        bad=[dict(c,planted_group=-1) for c in clients]
        self.assertIsNone(neighbor_recovery(d,bad,3)[0])
        m,_=discovery_metrics(np.ones((8,1)),d,bad)
        self.assertIsNone(m['planted_ari'])

    def test_k1_not_mistaken_for_recovery(self):
        clients=[dict(id=i,domain=i//2,planted_group=i%2) for i in range(8)]
        d=np.ones((8,8))-np.eye(8)
        m,_=discovery_metrics(np.ones((8,1)),d,clients)
        self.assertEqual(m['planted_ari'],0.)
        self.assertEqual(m['planted_nmi'],0.)
        self.assertAlmostEqual(m['domain_mi_nats'],0.)

    def test_finite_sampling_disjoint_balanced_reproducible(self):
        y=np.repeat(np.arange(10),1000)
        shards,g=allocate_finite(y,10,4,100,0,6,42)
        again,h=allocate_finite(y,10,4,100,0,6,42)
        self.assertEqual(shards,again);np.testing.assert_array_equal(g,h)
        self.assertEqual(len(sum(shards,[])),len(set(sum(shards,[]))))
        self.assertEqual(sum(g==0),2)
        self.assertTrue(any(not np.isclose((y[s]<5).mean(),.7 if group==0 else .3) for s,group in zip(shards,g)))

    def test_ledger_includes_auxiliary_and_multiple_models(self):
        ledger={};add(ledger,'discovery_models',10,20)
        merge_ledger(ledger,{'discovery_models':{'upload_bytes':10,'download_bytes':20}},'auxiliary_')
        add(ledger,'downstream_models',30,40);add(ledger,'downstream_models',30,40)
        self.assertEqual(totals(ledger),dict(upload_bytes=80,download_bytes=120,total_bytes=200))
        self.assertEqual(summary_bytes(2,10,512,'full'),2*(10*(512*4+8+4)+8))

    def test_grouping_rows_sum_to_one(self):
        x=np.array([[0,0],[.1,0],[4,4],[4.1,4]],float)
        d=np.linalg.norm(x[:,None]-x[None,:],axis=2)
        for method in ('fuzzy','agglomerative','spectral','kmeans'):
            u=assign(d,x,method,2,42,reference_core())
            np.testing.assert_allclose(u.sum(1),1.)
            self.assertTrue(np.isfinite(u).all())

    def test_equal_domain_mass(self):
        clients=[dict(domain=d,train=list(range(n))) for d,n in [(0,10),(0,30),(1,100),(1,100)]]
        np.testing.assert_allclose([domain_balanced_weight(c,clients) for c in clients],[.25,.75,.5,.5])


if __name__=='__main__':unittest.main()
