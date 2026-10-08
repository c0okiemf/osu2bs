import unittest
from eval.phrase_native_refine import select


class RefinementSelectionTests(unittest.TestCase):
    def test_native_fallback_is_distinct_from_production_fallback(self):
        template={'native':True};unknown={'ok':True,'machine':{'admitted':False}};admitted={'ok':True,'machine':{'admitted':True}}
        fallback=select([unknown],template,4)
        self.assertTrue(fallback['refinement_fallback']);self.assertFalse(fallback['fallback']);self.assertIs(fallback['selected'],template)
        self.assertEqual(fallback['template_seed'],4)
        chosen=select([unknown,admitted,admitted],template,4)
        self.assertEqual(chosen['selected_seed'],1);self.assertFalse(chosen['refinement_fallback']);self.assertIs(chosen['selected'],admitted)


if __name__=='__main__':unittest.main()
