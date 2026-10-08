"""A plateau must respect the requested stop policy and preserve snapshots."""
import json
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import torch
import groom


class ConstantFlow(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.bias=torch.nn.Parameter(torch.zeros(9+groom.DCOL+groom.DLAY+groom.MAX_CHAIN))

    def forward(self,x,*unused):
        v=self.bias.expand(*x.shape[:2],-1)
        return v.split((9,groom.DCOL,groom.DLAY,groom.MAX_CHAIN),dim=-1)


class FrozenOptimizer:
    def zero_grad(self):pass
    def step(self):pass


class BudgetTest(unittest.TestCase):
    def test_plateau_stop_and_full_budget_snapshot(self):
        torch.set_num_threads(2)
        n=groom.CTX+1
        sample=(torch.zeros(n,13),torch.zeros(n,2),[(i,0,0,0,0,1) for i in range(n)],
                torch.zeros(n),torch.zeros(n),1.0,'one-family')
        encoded=(torch.zeros(n,groom.NTOK),torch.zeros(n,4,dtype=torch.long))
        with tempfile.TemporaryDirectory() as td, patch.object(groom,'Flow',ConstantFlow), \
                patch.object(groom,'events_to_xy',return_value=encoded), \
                patch.object(torch.optim,'Adam',return_value=FrozenOptimizer()):
            out=Path(td)/'flow.pt'
            for patience,epochs,expected in [(30,34,32),(1,4,3),(None,4,4)]:
                h=groom.train_flow([sample],[sample],'cpu',random.Random(1),out_path=out,
                    max_epochs=epochs,steps_per_epoch=1,snapshot_updates=(2,),early_stop_patience=patience)
                self.assertEqual(h['updates'],expected)
                self.assertEqual(h['best_epoch'],0)
                self.assertTrue((out.parent/'flow-2.pt').exists())
                self.assertEqual(json.loads((out.parent/'history-2.json').read_text())['updates'],2)
            with self.assertRaises(ValueError):
                groom.train_flow([sample],[sample],'cpu',random.Random(1),early_stop_patience=-1)


if __name__=='__main__':unittest.main()
