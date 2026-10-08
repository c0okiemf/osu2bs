import unittest

from eval.local_corpus_recovery import audio_groups, training_pool


def test_pool_never_recovers_heldout_quarantine_or_existing_training():
    maps = [{'family': f, 'split': s} for f, s in
            [('new', 'train'), ('old', 'train'), ('val', 'train'), ('test', 'test'),
             ('qa', 'train'), ('q', 'train'), ('alias', 'train'), ('alias', 'dev')]]
    roles = {'old': {'future_role': 'train'}, 'val': {'future_role': 'validation'},
             'q': {'future_role': None}}
    pool, blocked, existing = training_pool(maps, roles, {'qa'})
    assert pool == ['new']
    assert blocked == {'val', 'test', 'qa', 'q', 'alias'}
    assert existing == {'old'}


def test_audio_overlap_blocks_transitively_and_groups_duplicates():
    groups = audio_groups(['new', 'bridge', 'test', 'train', 'copy'],
                          [('new', 'bridge'), ('bridge', 'test'), ('train', 'copy')], {'test'})
    assert groups['new']['blocked_by'] == ['test']
    assert groups['new']['group'] == groups['test']['group']
    assert groups['copy']['group'] == groups['train']['group']
    assert groups['copy']['blocked_by'] == []


class RecoveryTests(unittest.TestCase):
    def test_roles(self):
        test_pool_never_recovers_heldout_quarantine_or_existing_training()

    def test_overlap(self):
        test_audio_overlap_blocks_transitively_and_groups_duplicates()

    def test_short_clip_requires_full_containment_and_checks_late_audio(self):
        import numpy as np
        from eval.joint_audio_identity import peak_correlation
        rng = np.random.default_rng(1)
        clip = rng.normal(size=227)
        recording = rng.normal(size=10000)
        recording[7500:7727] = clip
        score = peak_correlation(recording, clip, max_lag=len(recording), min_overlap=len(clip))
        self.assertEqual(score['lag_samples'], 7500)
        self.assertEqual(score['overlap_samples'], len(clip))
        self.assertGreater(score['correlation'], .999999)
        partial = peak_correlation(clip[:-1], clip, max_lag=len(clip), min_overlap=len(clip))
        self.assertIsNone(partial['correlation'])


if __name__ == '__main__':
    unittest.main()
