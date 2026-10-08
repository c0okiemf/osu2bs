import unittest
from eval.joint_recovered_inventory import assign_roles


class RecoveredRoleTests(unittest.TestCase):
    def test_old_role_precedence_quarantine_and_order_independent_reservation(self):
        old={'a':'train','v':'validation','d':'development','bad':'train'}
        new=['new'+str(i) for i in range(10)];edges=[('bad','d'),('new0','v'),('new1','a'),('new2','bad')]
        result=assign_roles(old,new,edges,{'train':['bad'],'validation':[]})
        self.assertEqual(result,assign_roles(dict(reversed(list(old.items()))),list(reversed(new)),list(reversed(edges)),{'validation':[],'train':['bad']}))
        for f,role in (('new0','validation'),('new1','train'),('new2','development')):self.assertEqual(result['sources'][f]['future_role'],role)
        self.assertIsNone(result['sources']['bad']['future_role']);self.assertEqual(result['sources']['bad']['old_role'],'train')
        self.assertEqual(len(result['new_only_validation_groups']),5)
        self.assertFalse(set(result['new_only_validation_groups'])&{'new0','new1','new2'})
        with self.assertRaises(ValueError):assign_roles(old,['a'],[],{})


if __name__=='__main__':unittest.main()
