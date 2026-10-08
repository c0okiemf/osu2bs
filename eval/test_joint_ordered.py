import copy
import unittest
import numpy as np
from eval.joint_ordered import temporal_descriptor,ordered_bank,retrieve,select
from eval import joint_pilot as old


class OrderedTests(unittest.TestCase):
    def test_temporal_order_selects_corresponding_literal_donor(self):
        times=np.arange(0,4,.01);activity=(times<2).astype(float)
        a={'times':times,'x':np.tile(activity[:,None],(1,27))}
        b={'times':times,'x':a['x'][::-1].copy()}
        np.testing.assert_array_equal(old.phrase_descriptor(a,0,8,120,1),old.phrase_descriptor(b,0,8,120,1))
        self.assertFalse(np.array_equal(temporal_descriptor(a,0,8,120),temporal_descriptor(b,0,8,120)))
        ds=[]
        for family,audio,beat in (('a',a,1.),('b',b,5.)):
            source={'bpm':120.,'duration_beats':8.,'notes':[(beat,0,0,0,1)],'bombs':[],'walls':[],
                    'windows':[{'start':0.,'end':8.,'events':[{'beat':beat,'hands':[[(0,0,1)],[]]}]}]}
            ds.append({'family':family,'fit_role':'train','source':source,'audio':audio})
        base=old.retrieval_bank(ds);bank=ordered_bank(ds,base)
        src={'bpm':120.,'duration_beats':8.,'notes':[],'events':[],'bombs':[],'walls':[]}
        # Equal old descriptors tie stably; ordered matching resolves opposite halves.
        self.assertEqual(old.retrieve_song(src,b,base,.25)['donors'],[('a',0.)])
        self.assertEqual(retrieve(src,b,bank,.25)['donors'],[('b',0.)])
        self.assertEqual(retrieve(src,a,bank,.25)['source']['notes'],[(1.,0,0,0,1)])
        invalid=copy.deepcopy(ds);invalid[0]['fit_role']='validation'
        with self.assertRaisesRegex(ValueError,'train-only'):ordered_bank(invalid,base)

    def test_tail_bins_and_admission_fallback(self):
        audio={'times':np.arange(0,4,.01),'x':np.ones((400,27))}
        desc=temporal_descriptor(audio,0,1,120).reshape(16,2)
        np.testing.assert_array_equal(desc[:2],np.ones((2,2)));np.testing.assert_array_equal(desc[2:],np.zeros((14,2)))
        baseline={'b0':True};rejected={'ok':True,'machine':{'admitted':False}}
        self.assertIs(select([rejected],baseline)['selected'],baseline)
        admitted={'ok':True,'machine':{'admitted':True}}
        self.assertEqual(select([rejected,admitted,admitted],baseline)['selected_seed'],1)


if __name__=='__main__':unittest.main()
