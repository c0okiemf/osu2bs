import unittest
import numpy as np
from eval.phrase_native import counts,donor_order,render


class NativeBudgetTests(unittest.TestCase):
    def test_event_counts_and_lexicographic_budget_priority(self):
        notes=[(0.,0,0,0,0),(0.,0,1,0,8),(0.,1,3,0,1),(1.,1,2,1,8)]
        self.assertEqual(counts(notes),[0,1,1])
        order,cost=donor_order([[0,1,1],[0,2,1],[0,1,1]],[0,1,1],[10.,0.,2.])
        np.testing.assert_array_equal(order,[2,0,1]);np.testing.assert_array_equal(cost,[0,1,0])

    def test_rest_preserves_history_and_tail_counts_literal_events(self):
        def entry(family,notes):
            return {'family':family,'start':0,'notes':notes,'entry_gap_beats':[.25,.25],'exit_gap_beats':[.25,.25]}
        a=entry('a',[(0.,0,0,0,0),(2.,0,1,0,8)])
        rest=entry('rest',[]);b=entry('b',[(.5,1,3,0,1),(2.,1,2,1,8)])
        bank={'entries':[a,rest,b],'mean':np.zeros(55),'sd':np.ones(55),'x':np.zeros((3,55))}
        source={'duration_beats':17.,'bpm':120.,'notes':[],'bombs':[],'walls':[]}
        plans=[{'start':0.,'end':8.,'counts':[2,0,0]},{'start':8.,'end':16.,'counts':[0,0,0]},
               {'start':16.,'end':17.,'counts':[0,1,0]}]
        audio={'times':np.arange(10.),'x':np.zeros((10,27))}
        result=render(source,audio,bank,plans,1.,0)
        self.assertTrue(result['ok']);ledger=result['budget_ledger']
        self.assertEqual([p['absolute_count_error'] for p in ledger],[0,0,0])
        self.assertEqual(ledger[1]['prior_hands'],ledger[2]['prior_hands'])
        self.assertEqual(ledger[2]['prior_hands'][0]['beat'],2.)
        self.assertEqual(result['source']['notes'],a['notes']+[(16.5,1,3,0,1)])


if __name__=='__main__':unittest.main()
