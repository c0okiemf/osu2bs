import copy
import unittest
from eval.hand_state_geometry_fit import choose,qualify


class SelectionTests(unittest.TestCase):
    def test_complete_earliest_rollout_and_frozen_margin(self):
        control={'attempts':12,'failures':0,'geometry':{'mean':10.,'components':[10.]*5,'supported_windows':[10]*5}}
        selected={**copy.deepcopy(control),'update':1000};selected['geometry']['mean']=9.
        records=[{'signature_ok':True,'phase_signature_ok':True,'direction_counters_ok':True}]*12
        self.assertTrue(all(qualify(selected,control,records).values()))
        other={**copy.deepcopy(selected),'update':3000};incomplete={**copy.deepcopy(selected),'failures':1,'update':0}
        incomplete['geometry']['mean']=0.
        self.assertEqual(choose([other,incomplete,selected])['update'],1000)
        selected['geometry']['mean']=9.1
        self.assertFalse(qualify(selected,control,records)['geometry_10percent_vs_control'])
        selected['geometry']['components'][0]=None;selected['geometry']['supported_windows'][1]=9
        gates=qualify(selected,control,records)
        self.assertFalse(gates['geometry_components_vs_control']);self.assertFalse(gates['geometry_coverage_vs_control'])


if __name__=='__main__':unittest.main()
