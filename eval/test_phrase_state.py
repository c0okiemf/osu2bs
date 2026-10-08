from copy import deepcopy
import unittest
from eval.phrase_state import encode,decode,canonical


def source():
    return {'bpm':120.,'duration_beats':25.25,'authored':{'njs':18.,'offset_beats':0.},
        'bombs':[(9.,2,1)],'walls':[(7.,3.,0,1,0,5)],
        'notes':[(0.,0,0,0,0),(0.,0,0,1,8),(0.,1,3,0,1),(.00000001,1,3,1,0),
                 (16.,1,2,1,1),(25.,0,1,2,4)]}


class PhraseStateTests(unittest.TestCase):
    def test_exact_scene_and_history_across_empty_windows_and_micro_offsets(self):
        original=source();packet=canonical(encode(original));actual=decode(packet)
        for key in ('notes','bombs','walls'):self.assertEqual(list(map(tuple,actual[key])),original[key])
        self.assertEqual(packet['plans'][0]['counts'],[0,1,1]);self.assertEqual(packet['plans'][1]['counts'],[0,0,0])
        self.assertEqual(packet['plans'][-1]['end'],25.25)
        at16=next(s for s in packet['steps'] if s['action']['gap'] and s['action']['target_beat']==16.)
        self.assertEqual(at16['state']['hands'][0],{'beat':0.,'notes':[[0,0,0],[0,1,8]]})
        self.assertEqual(at16['state']['age_beats'][0],16.)

    def test_future_geometry_does_not_change_budgets_or_earlier_history(self):
        before=source();after=deepcopy(before);after['notes'][-1]=(25.,0,2,0,7)
        a,b=encode(before),encode(after);self.assertEqual(a['plans'],b['plans'])
        for x,y in zip(a['steps'],b['steps']):
            if x['action']['target_beat']<=25.:self.assertEqual(x['state'],y['state'])

    def test_altered_counts_history_notes_or_missing_rest_are_rejected(self):
        original=canonical(encode(source()))
        for mode in ('budget','history','note','missing_rest','extra_action'):
            packet=deepcopy(original)
            if mode=='budget':packet['plans'][0]['counts'][0]+=1
            if mode=='history':packet['steps'][1]['state']['hands'][0]['notes'][0][2]=7
            if mode=='note':packet['steps'][0]['action']['slots'][0]=2
            if mode=='missing_rest':packet['steps'].pop()
            if mode=='extra_action':packet['steps'].append(deepcopy(packet['steps'][-1]))
            with self.assertRaises(ValueError,msg=mode):decode(packet)


if __name__=='__main__':unittest.main()
