import unittest
import numpy as np

from eval.corpus_quick_wins import family_weights, validate_additions, expanded_bank
from eval.joint_phrase import phrase_windows


class CorpusQuickWinsTests(unittest.TestCase):
    def test_family_mass_is_independent_of_chart_length(self):
        families = ['short'] * 2 + ['long'] * 9
        weights = family_weights(families)
        self.assertAlmostEqual(weights[:2].sum(), weights[2:].sum())
        self.assertAlmostEqual(weights[:2].sum(), 1.)

    def test_additions_reject_heldout_roles_aliases_and_duplicates(self):
        row = {'family': 'new', 'role': 'train',
               'audio_group': {'members': ['new'], 'blocked_by': []}}
        validate_additions([row], {'old'}, {'val'})
        for bad in ({**row, 'role': 'validation'}, {**row, 'family': 'old'},
                    {**row, 'audio_group': {'members': ['new', 'val'], 'blocked_by': []}}):
            with self.assertRaises(ValueError):
                validate_additions([bad], {'old'}, {'val'})
        with self.assertRaises(ValueError):
            validate_additions([row, row], {'old'}, {'val'})

    def test_bank_extension_keeps_old_normalization_and_rejects_unapproved(self):
        mean = np.arange(55, dtype=float); sd = np.ones(55) * 2
        base = {'entries': [{'family': 'old', 'start': 0., 'notes': [], 'descriptor': mean.copy()}],
                'mean': mean.copy(), 'sd': sd.copy(), 'x': np.zeros((1, 55))}
        source = {'bpm': 120., 'duration_beats': 16., 'notes': [(2., 0, 0, 0, 1), (10., 0, 0, 1, 0)]}
        source['windows'] = phrase_windows(source['notes'], [], [], 16.)
        audio = {'times': np.arange(0, 8, .1), 'x': np.zeros((80, 27))}
        row = {'family': 'new', 'fit_role': 'train', 'approved': True, 'source': source, 'audio': audio}
        bank = expanded_bank(base, [row])
        np.testing.assert_array_equal(bank['mean'], mean)
        np.testing.assert_array_equal(bank['sd'], sd)
        np.testing.assert_array_equal(bank['x'][0], base['x'][0])
        self.assertIs(bank['entries'][0], base['entries'][0])
        self.assertEqual(len(bank['entries']), 3)
        self.assertEqual(bank['entries'][1]['descriptor'][-1], 1 / 8)
        self.assertEqual(bank['entries'][1]['exit_gap_beats'], [8., None])
        with self.assertRaises(ValueError):
            expanded_bank(base, [{**row, 'approved': False}])

    def test_export_diagnostics_serialize_numpy_derived_counts(self):
        import json
        import tempfile
        from pathlib import Path
        from eval.corpus_quick_wins import diagnostics
        from eval.joint_export import export_chart
        source = {'bpm': 120., 'duration_beats': 8., 'walls': [], 'bombs': [],
                  'notes': [(1., 0, 0, 0, 1), (1.001, 0, 0, 1, 0), (4., 0, 0, 0, 2), (4., 1, 3, 0, 3)]}
        bank = {'entries': [{'family': 'donor', 'start': 0., 'notes': source['notes'],
                            'entry_gap_beats': [None, None], 'exit_gap_beats': [None, None]}]}
        profile = {'rhythm': [[.5, .5, 0., 0., 0.]]}
        record = {'ok': True, 'donors': [['donor', 0.]], 'profile': profile,
                  'reference_profile': profile, 'budget_ledger': [{'absolute_count_error': 0}]}
        item = {'source': source, 'rate': .5, 'data': {'source': source}}
        with tempfile.TemporaryDirectory() as d:
            export_chart(source, d)
            result = diagnostics(record, Path(d), item, bank, {'donor'})
        saved = json.loads(json.dumps(result))
        self.assertEqual(saved['submillisecond_hand_gaps'], 1)
        self.assertEqual(saved['horizontal_pairs'], {'narrow': 1, 'broad': 1, 'converging': 0, 'opposite_horizontal': 1})


if __name__ == '__main__':
    unittest.main()
