import unittest
from eval.phrase_native_arc_origin import window


class ArcOriginTests(unittest.TestCase):
    def test_donor_origin_dots_and_simultaneous_heads(self):
        notes=[(0.,0,0,0,0),(.2,0,1,0,1),(.4,0,2,1,0),(.6,0,2,2,1)]
        a=window(notes,120.,{n:0 for n in notes});self.assertEqual(a['counts']['legacy'],1)
        self.assertEqual(a['counts']['unambiguous_without_seams'],1)
        b=window(notes,120.,{n:int(i>=2) for i,n in enumerate(notes)})
        self.assertEqual(b['counts']['unambiguous'],1);self.assertEqual(b['counts']['unambiguous_without_seams'],0)
        same=window(sorted(notes+[(.3,0,3,0,8)]),120.);other=window(sorted(notes+[(.3,1,3,0,8)]),120.)
        self.assertEqual(same['counts']['dot_split'],0);self.assertEqual(other['counts']['dot_split'],1)
        stack=[notes[0],notes[1],(.2,*notes[2][1:]),notes[3]]
        c=window(stack,120.);self.assertEqual(c['counts']['legacy'],1);self.assertEqual(c['counts']['unambiguous'],0)
        self.assertTrue(c['runs'][0]['simultaneous_multi_head'])


if __name__=='__main__':unittest.main()
