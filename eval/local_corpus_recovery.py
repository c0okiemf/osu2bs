"""Cheap additive training inventory from existing files; no fit or old-manifest edits.

python -m eval.local_corpus_recovery [verify]
"""
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

import numpy as np

from eval.corpus import _fingerprint, FP_CORR
from eval.expressive_manifest import APPROVED_ROOTS, family_strata, freeze_run
from eval.fresh_confirm import _excluded
from eval.joint_audio_identity import peak_correlation
from eval.joint_export import export_chart, read_chart, assert_same
from eval.joint_phrase import ROOT, PROTECTED, UnsupportedSource, _sha, _atomic, read_source

OUT = ROOT / 'experiments/local-corpus-recovery-v1'
MANIFEST = ROOT / 'eval/corpus_manifest.json'
INVENTORY = ROOT / 'experiments/joint-recovered-inventory-v1/inventory.json'


def training_pool(maps, roles, excluded):
    blocked = set(excluded)
    blocked.update(m['family'] for m in maps if m['split'] != 'train')
    blocked.update(f for f, r in roles.items() if r['future_role'] != 'train')
    existing = {f for f, r in roles.items() if r['future_role'] == 'train'}
    return sorted({m['family'] for m in maps} - blocked - existing), blocked, existing


def audio_groups(families, edges, blocked):
    parent = {f: f for f in families}
    def find(f):
        while parent[f] != f:
            parent[f] = parent[parent[f]]
            f = parent[f]
        return f
    for a, b in edges:
        a, b = find(a), find(b)
        parent[max(a, b)] = min(a, b)
    groups = defaultdict(list)
    for f in sorted(parent):
        groups[find(f)].append(f)
    return {f: {'group': g, 'members': members,
                'blocked_by': sorted(set(members) & set(blocked))}
            for g, members in groups.items() for f in members}


def freeze():
    corpus = json.loads(MANIFEST.read_text())
    inventory = json.loads(INVENTORY.read_text())
    strata = family_strata(corpus)
    excluded = set(strata['rejected']) | set().union(*_excluded().values())
    pool, blocked, existing = training_pool(corpus['maps'], inventory['roles']['sources'], excluded)
    candidates, audio = [], {}
    for m in sorted(corpus['maps'], key=lambda r: r['dir']):
        directory = Path(m['dir'])
        ap = directory / m['audio_file'] if m.get('audio_file') else None
        if ap is not None and ap.is_file():
            h = _sha(ap)
            item = audio.setdefault(h, {'path': str(ap), 'sha256': h, 'families': []})
            if m['family'] not in item['families']:
                item['families'].append(m['family'])
        if m['family'] not in pool:
            continue
        ep = m['charts'].get('ExpertPlus')
        infos = sorted(p for p in directory.iterdir() if p.name.lower() == 'info.dat')
        cp = directory / ep['file'] if ep else None
        row = {'family': m['family'], 'dir': m['dir'], 'legacy_eligibility': m['eligible'],
               'mapper': m['level_author'], 'approved_chart': any(p in m['dir'] for p in APPROVED_ROOTS),
               'sources': {}, 'inventory_reason': None}
        if cp is None or not cp.is_file() or ap is None or not ap.is_file() or len(infos) != 1:
            row['inventory_reason'] = 'missing_chart_audio_or_unique_info'
        else:
            row['sources'] = {k: {'path': str(p), 'sha256': _sha(p)}
                              for k, p in [('chart', cp), ('info', infos[0]), ('audio', ap)]}
            if row['sources']['chart']['sha256'] != ep['sha256']:
                raise ValueError('original chart identity changed')
            info = json.loads(infos[0].read_text(encoding='utf-8-sig'))
            if info.get('_songFilename') != ap.name:
                row['inventory_reason'] = 'audio_filename_binding_mismatch'
        candidates.append(row)
    inputs = [MANIFEST, INVENTORY, ROOT / 'experiments/joint-recovered-inventory-v1/report.json']
    dependencies = [__file__, 'eval/corpus.py', 'eval/joint_audio_identity.py', 'eval/joint_phrase.py',
                    'eval/joint_export.py', 'eval/map_reader.py', 'eval/map_scope.py', 'eval/visibility.py',
                    'eval/expressive_manifest.py', 'eval/fresh_confirm.py', 'eval/e3_turning.py']
    return freeze_run(OUT, {'candidates': candidates, 'audio': audio, 'pool': pool,
        'blocked': sorted(blocked), 'existing_train': sorted(existing),
        'prior_audio_groups': inventory['roles']['groups'],
        'inputs': {str(p): _sha(p) for p in inputs},
        'dependencies': {str(p): _sha(ROOT / p) for p in dependencies},
        'protected': {p: _sha(ROOT / p) for p in PROTECTED},
        'policy': {'min_notes': 100, 'min_span_s': 30, 'quarter_grid_required': False,
                   'audio_seconds': 90, 'audio_lag_s': 30, 'audio_overlap_s': 30,
                   'audio_threshold': FP_CORR, 'no_new_fit': True}})


def run():
    frozen = freeze(); config = frozen['config']; supported = {}; attempts = []
    for row in config['candidates']:
        if row['family'] in supported:
            continue
        result = {'record': row, 'reason': row['inventory_reason']}
        if not result['reason']:
            try:
                source = read_source(row)
                if len(source['notes']) < 100 or (source['notes'][-1][0] - source['notes'][0][0]) * 60 / source['bpm'] < 30:
                    raise UnsupportedSource('too_few_notes_or_short_span')
                dest = OUT / 'exports' / row['family'].replace(':', '_')
                njs, offset = source['authored']['njs'], source['authored']['offset_beats']
                assert_same(source, export_chart(source, dest, njs, offset), njs, offset)
                payload = OUT / 'sources' / (row['family'].replace(':', '_') + '.json')
                _atomic(payload, source)
                result.update(payload=str(payload), payload_sha256=_sha(payload),
                    exports={str(dest / p): _sha(dest / p) for p in ('ExpertPlus.dat', 'Info.dat')},
                    notes=len(source['notes']), events=len(source['events']), windows=len(source['windows']),
                    event_rate=len(source['events']) / (source['duration_beats'] * 60 / source['bpm']))
                supported[row['family']] = result
            except UnsupportedSource as exc:
                result['reason'] = str(exc)
        attempts.append(result)
    _atomic(OUT / 'source-readiness.json', {'identity': frozen['identity'], 'attempts': attempts})
    print('supported local additions before audio screen:', len(supported), flush=True)
    envelopes = {}; unavailable = []
    for i, (h, item) in enumerate(config['audio'].items()):
        try:
            env = _fingerprint(Path(item['path']))
            if len(env) < 1500 or not np.isfinite(env).all() or np.std(env) < 1e-12:
                raise ValueError('short_or_constant_audio')
            envelopes[h] = env
        except (ValueError, RuntimeError) as exc:
            unavailable.append({'sha256': h, 'families': item['families'], 'reason': str(exc)})
        if i % 100 == 0:
            print('audio decoded', i + 1, '/', len(config['audio']), flush=True)
    edges = []
    # Retain previously corroborated transitive groups, including quarantines.
    for group in config['prior_audio_groups']:
        edges.extend((group['families'][0], f) for f in group['families'][1:])
    for item in config['audio'].values():
        edges.extend((item['families'][0], f) for f in item['families'][1:])
    checked = set(); matches = []; screen_unknown = set()
    for family, row in supported.items():
        h = row['record']['sources']['audio']['sha256']
        if h not in envelopes:
            screen_unknown.add(family); continue
        for other, item in config['audio'].items():
            if other == h or other not in envelopes:
                continue
            pair = tuple(sorted((h, other)))
            if pair in checked:
                continue
            checked.add(pair)
            score = peak_correlation(envelopes[h], envelopes[other])
            if score['correlation'] is not None and score['correlation'] >= FP_CORR:
                matches.append({'audio_sha256': [h, other], **score})
                edges.extend((a, b) for a in config['audio'][h]['families'] for b in item['families'])
    families = set(config['pool']) | set(config['existing_train']) | set(config['blocked'])
    groups = audio_groups(families, edges, config['blocked'])
    admitted = []; quarantine = []
    for family, row in sorted(supported.items()):
        item = {**row, 'family': family, 'role': 'train', 'audio_group': groups[family]}
        if groups[family]['blocked_by'] or family in screen_unknown:
            item['role'] = None; quarantine.append(item)
        else:
            admitted.append(item)
    # A failed reference decode leaves screening incomplete: do not claim admission.
    status = 'LOCAL_CORPUS_READY' if not unavailable else 'SOURCE_READY_AUDIO_SCREEN_INCOMPLETE'
    if unavailable:
        quarantine.extend({**r, 'role': None, 'reason': 'reference_audio_screen_incomplete'} for r in admitted)
        admitted = []
    old = set(config['existing_train'])
    new_groups = {r['audio_group']['group'] for r in admitted if not set(r['audio_group']['members']) & old}
    report = {'identity': frozen['identity'], 'status': status, 'sources': admitted, 'quarantined': quarantine,
        'summary': {'candidate_families': len(config['pool']), 'supported_families': len(supported),
                    'added_families': len(admitted), 'new_audio_groups': len(new_groups),
                    'future_train_families': len(old) + len(admitted), 'added_notes': sum(r['notes'] for r in admitted),
                    'added_events': sum(r['events'] for r in admitted), 'added_windows': sum(r['windows'] for r in admitted),
                    'approved_charts': sum(r['record']['approved_chart'] for r in admitted),
                    'legacy_eligibility': dict(Counter(r['record']['legacy_eligibility'] for r in admitted)),
                    'mappers': sorted({r['record']['mapper'] or '(unknown)' for r in admitted}),
                    'event_rate_min_median_max': np.quantile([r['event_rate'] for r in admitted], [0, .5, 1]).tolist() if admitted else [],
                    'failed_attempt_reasons': dict(Counter(r['reason'] for r in attempts if r['reason']))},
        'audio': {'unique_recordings': len(config['audio']), 'pairs_screened': len(checked),
                  'matches': matches, 'unavailable': unavailable},
        'claim': 'Additive future training sources only. Sample audio groups uniformly to avoid duplicate mass. '
                 'Existing 90-second envelope screen is conservative, not exhaustive song identity. '
                 'Unlabeled maps are not newly approved; no training, evaluation promotion or flow-quality claim.'}
    _atomic(OUT / 'manifest.json', report)
    verify()
    return report['summary']


def verify():
    run = json.loads((OUT / 'run.json').read_text()); config = run['config']
    manifest = json.loads((OUT / 'manifest.json').read_text())
    if manifest['identity'] != run['identity']:
        raise ValueError('manifest identity mismatch')
    for path, h in {**config['inputs'], **config['dependencies'], **config['protected']}.items():
        if _sha(ROOT / path) != h:
            raise ValueError('protected/input/code identity changed: ' + path)
    for row in manifest['sources']:
        assert row['role'] == 'train' and not row['audio_group']['blocked_by']
        assert row['family'] not in config['blocked'] and row['family'] not in config['existing_train']
        source = read_source(row['record'])
        if _sha(row['payload']) != row['payload_sha256']:
            raise ValueError('payload changed')
        for p, h in row['exports'].items():
            if _sha(p) != h:
                raise ValueError('export changed')
        dest = OUT / 'exports' / row['family'].replace(':', '_')
        assert_same(source, read_chart(dest / 'ExpertPlus.dat', dest / 'Info.dat'),
                    source['authored']['njs'], source['authored']['offset_beats'])
    result = {'identity': run['identity'], 'manifest_sha256': _sha(OUT / 'manifest.json'),
              'verified_sources': len(manifest['sources']), 'old_artifacts_unchanged': True}
    _atomic(OUT / 'verification.json', result)
    return result


if __name__ == '__main__':
    print(json.dumps(verify() if sys.argv[1:] == ['verify'] else run(), indent=1))
