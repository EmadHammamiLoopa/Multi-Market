import json
import unittest
import numpy as np

from multimarket.v23_phase0dj_score import _prior_z, _greedy, _gross, _align_state_decisions, _gate


class Phase0DJScoreTests(unittest.TestCase):
    def test_prior_z_excludes_current_value(self):
        x=np.asarray([1.0,2.0,3.0,4.0])
        z=_prior_z(x,w=3)
        self.assertTrue(np.isnan(z[0]))
        self.assertTrue(np.isnan(z[1]))
        self.assertTrue(np.isnan(z[2]))
        mu=2.0
        sd=np.std(np.asarray([1.0,2.0,3.0]),ddof=0)
        self.assertAlmostEqual(z[3],(4.0-mu)/sd)

    def test_greedy_prevents_same_symbol_overlap(self):
        ix=np.asarray([0,1,4,5,6,10,11],dtype=np.int64)
        self.assertEqual(_greedy(ix,5).tolist(),[0,6])

    def test_gross_uses_exact_future_minute_row(self):
        p=np.asarray([100.0,101.0,102.0,103.0])
        g=_gross(p,2)
        self.assertAlmostEqual(g[0],np.log(102.0/100.0)*10000.0)
        self.assertTrue(np.isnan(g[-1]))

    def test_alignment_trims_only_outside_trade_boundaries(self):
        state_open=np.asarray([0,60,120,180,240],dtype=np.int64)
        ts=np.arange(100,260,dtype=np.int64)
        keep,idx,decision=_align_state_decisions(ts,state_open)
        self.assertEqual(keep.tolist(),[False,True,True,True,False])
        self.assertEqual(decision.tolist(),[119,179,239])
        self.assertEqual(ts[idx].tolist(),[119,179,239])

    def test_alignment_rejects_missing_interior_decision_second(self):
        state_open=np.asarray([0,60,120,180,240],dtype=np.int64)
        ts=np.arange(0,300,dtype=np.int64)
        ts=ts[ts!=179]
        with self.assertRaisesRegex(ValueError,'contiguous inside DEV range'):
            _align_state_decisions(ts,state_open)

    def test_gate_returns_builtin_bool_and_is_json_serializable(self):
        p12={
            'fold_expectancies':[1.0,1.0,1.0,1.0,1.0],
            'positive_outer_folds':5,
            'net_bps_trade':2.0,
            'total_net_bps':100.0,
            'profit_factor':1.5,
            'positive_active_day_fraction':0.8,
            'pnl_to_drawdown':3.0,
            'median_trades_day_active':3.0,
        }
        p15={
            'net_bps_trade':1.0,
            'total_net_bps':50.0,
        }
        value=_gate(p12,p15)
        self.assertIs(type(value),bool)
        self.assertEqual(json.dumps({'candidate_structural_gate':value}),'{"candidate_structural_gate": true}')


if __name__=='__main__':
    unittest.main()
