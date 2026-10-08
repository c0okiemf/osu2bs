import unittest
from eval.joint_generalization_audit import spans


class TeacherAuditTests(unittest.TestCase):
    def test_every_target_is_scored_once_with_real_start_and_nonbos_warmup(self):
        for length in (1,32,70,127,128,129,223,224,225,1307):
            for cropped in (False,True):
                selected=[]
                for start,end,warmup in spans(length,cropped):
                    self.assertEqual(warmup,32 if start else 0)
                    selected.extend(range(start+warmup,end))
                self.assertEqual(selected,list(range(length)))


if __name__=='__main__':unittest.main()
