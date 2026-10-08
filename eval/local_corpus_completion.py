"""Complete local recovery with full-clip checks for short held-out recordings."""
import json
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf

from eval import local_corpus_recovery as parent
from eval.corpus import _fingerprint, FP_CORR
from eval.expressive_manifest import freeze_run
from eval.joint_audio_identity import peak_correlation
from eval.joint_export import assert_same, read_chart
from eval.joint_phrase import ROOT, _sha, _atomic, read_source

OUT = ROOT / 'experiments/local-corpus-recovery-v2'


def run():
    parent.verify()
    prior = json.loads((parent.OUT / 'manifest.json').read_text())
    config = json.loads((parent.OUT / 'run.json').read_text())['config']
    if prior['status'] != 'SOURCE_READY_AUDIO_SCREEN_INCOMPLETE':
        raise ValueError('expected incomplete parent')
    paths = [parent.OUT / p for p in ('run.json', 'manifest.json', 'verification.json')]
    frozen = freeze_run(OUT, {'parent_inputs': {str(p): _sha(p) for p in paths},
        'code_sha256': _sha(__file__), 'policy': 'Every unavailable recording must be nonconstant and shorter than 30s. '
        'Screen its complete RMS envelope anywhere in each complete candidate recording; '
        'no partial-clip matches, correlation >=0.95. Quarantine matches. No source or old outcome changes.'})
    short = {}
    for item in prior['audio']['unavailable']:
        recording = config['audio'][item['sha256']]
        if _sha(recording['path']) != item['sha256']:
            raise ValueError('short reference identity changed')
        duration = sf.info(recording['path']).duration
        if not 0 < duration < 30:
            raise ValueError('unresolved audio decode or constant long reference')
        env = _fingerprint(Path(recording['path']))
        if not np.isfinite(env).all() or np.std(env) < 1e-12:
            raise ValueError('unresolved constant short reference')
        short[item['sha256']] = (env, duration, recording['families'])
    admitted, quarantined, scores = [], [], []
    for row in prior['quarantined']:
        if row['audio_group']['blocked_by']:
            quarantined.append(row); continue
        if row.get('reason') != 'reference_audio_screen_incomplete':
            raise ValueError('unexpected quarantine reason')
        source = read_source(row['record'])
        if _sha(row['payload']) != row['payload_sha256']:
            raise ValueError('payload changed')
        for path, h in row['exports'].items():
            if _sha(path) != h:
                raise ValueError('export changed')
        dest = parent.OUT / 'exports' / row['family'].replace(':', '_')
        assert_same(source, read_chart(dest / 'ExpertPlus.dat', dest / 'Info.dat'),
                    source['authored']['njs'], source['authored']['offset_beats'])
        path = Path(row['record']['sources']['audio']['path'])
        duration = sf.info(path).duration
        candidate = _fingerprint(path, secs=duration)
        blocked = []
        for h, (env, seconds, families) in short.items():
            score = peak_correlation(candidate, env, max_lag=len(candidate), min_overlap=len(env))
            if score['correlation'] is None:
                raise ValueError('short-reference comparison unsupported')
            scores.append({'family': row['family'], 'reference_sha256': h,
                           'reference_duration_s': seconds, **score})
            if score['correlation'] >= FP_CORR:
                blocked.extend(families)
        item = {**row, 'role': None if blocked else 'train', 'reason': 'short_clip_overlap' if blocked else None,
                'short_clip_blocked_by': sorted(set(blocked))}
        (quarantined if blocked else admitted).append(item)
    # A match removes the complete existing audio group from these additions.
    bad_groups = {r['audio_group']['group'] for r in quarantined}
    for row in list(admitted):
        if row['audio_group']['group'] in bad_groups:
            admitted.remove(row); quarantined.append({**row, 'role': None, 'reason': 'group_short_clip_overlap'})
    old = set(config['existing_train'])
    summary = {'previous_train_families': len(old), 'added_families': len(admitted),
        'future_train_families': len(old) + len(admitted),
        'new_audio_groups': len({r['audio_group']['group'] for r in admitted if not set(r['audio_group']['members']) & old}),
        'added_notes': sum(r['notes'] for r in admitted), 'added_events': sum(r['events'] for r in admitted),
        'added_windows': sum(r['windows'] for r in admitted),
        'approved_charts': sum(r['record']['approved_chart'] for r in admitted),
        'legacy_eligibility': dict(Counter(r['record']['legacy_eligibility'] for r in admitted)),
        'mapper_credits': sorted({r['record']['mapper'] or '(unknown)' for r in admitted}),
        'event_rate_min_median_max': np.quantile([r['event_rate'] for r in admitted], [0, .5, 1]).tolist() if admitted else []}
    report = {'identity': frozen['identity'], 'status': 'LOCAL_CORPUS_READY', 'summary': summary,
        'sources': admitted, 'quarantined': quarantined, 'short_clip_scores': scores,
        'parent_manifest_sha256': _sha(parent.OUT / 'manifest.json'), 'claim': prior['claim']}
    _atomic(OUT / 'manifest.json', report)
    parent.verify()
    _atomic(OUT / 'verification.json', {'identity': frozen['identity'], 'manifest_sha256': _sha(OUT / 'manifest.json'),
        'verified_sources': len(admitted), 'old_artifacts_unchanged': True, 'short_clip_comparisons': len(scores)})
    return summary


if __name__ == '__main__':
    print(json.dumps(run(), indent=1))
