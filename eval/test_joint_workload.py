import unittest
import numpy as np
from eval import joint_ordered as ordered,joint_pilot as old
from eval.joint_workload import corrected_bank,retrieve
from eval.joint_phrase import encode_events


class WorkloadTests(unittest.TestCase):
    def test_equal_donor_second_rates_choose_actual_target_workload(self):
        train=[]
        for fam,bpm,n in (('a',60.,32),('b',120.,16)):
            times=np.arange(0,8*60/bpm,.01);audio={'times':times,'x':np.zeros((len(times),27))}
            notes=[(i*8/n,0,0,0,i%2) for i in range(n)]
            source={'bpm':bpm,'duration_beats':8.,'notes':notes,'bombs':[],'walls':[],
                    'windows':[{'start':0.,'end':8.,'events':encode_events(notes)}]}
            train.append({'family':fam,'fit_role':'train','source':source,'audio':audio})
        base=ordered.ordered_bank(train,old.retrieval_bank(train));bank=corrected_bank(base)
        source={'bpm':120.,'duration_beats':8.,'notes':[],'events':[],'walls':[],'bombs':[]};audio=train[1]['audio']
        old_candidate=old.retrieve_song(source,audio,base,4.)
        candidate=retrieve(source,audio,bank,4.)
        self.assertEqual(old_candidate['donors'],[('a',0.)]);self.assertEqual(candidate['donors'],[('b',0.)])
        self.assertEqual(len(old_candidate['source']['notes'])/4,8.)
        self.assertEqual(len(candidate['source']['notes'])/4,4.)
        altered={**bank,'temporal_x':bank['temporal_x']+1000}
        self.assertEqual(retrieve(source,audio,altered,4.)['source']['notes'],candidate['source']['notes'])
        for k in ('mean','sd'):
            np.testing.assert_array_equal(base[k][:-1],bank[k][:-1])
        np.testing.assert_array_equal(base['x'][:,:54],bank['x'][:,:54])
        for k in ('temporal_mean','temporal_sd','temporal_x'):np.testing.assert_array_equal(base[k],bank[k])
        self.assertEqual([e['descriptor'][-1] for e in base['entries']],[4.,4.])
        self.assertEqual([e['descriptor'][-1] for e in bank['entries']],[4.,2.])
        with self.assertRaises(ValueError):retrieve(source,audio,bank,-1.)


if __name__=='__main__':unittest.main()
