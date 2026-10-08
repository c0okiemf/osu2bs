"""Two isolated corpus ablations. No production writes or automatic promotion.

python -m eval.corpus_quick_wins all [--out experiments/corpus-quick-wins-2026-10-02]
Stages: prepare, fit, evaluate, report. Use review to remeasure existing exports
after a reporting-only fix, with separate generation/measurement code identities.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import time

import numpy as np
import torch

from eval import phrase_budget as budget, phrase_native as native
from eval import joint_pilot as old, joint_ordered as ordered, joint_uniform_audio as uniform
from eval import joint_approved_fit as approved, joint_compatible as compatible, joint_joins as joins
from eval import joint_workload as workload
from eval.joint_audio_view import render_common
from eval.joint_continuity import describe
from motion import lr_doubles
from eval.quality_metrics import opposite_horizontal
from eval.joint_deployment import deployment_input, reference_view
from eval.joint_export import read_chart
from eval.joint_model import audio_features, _torch_save
from eval.joint_phrase import ROOT, PROTECTED, _sha, _atomic, verify_sources
from eval.expressive_manifest import APPROVED_ROOTS, freeze_run

OUT = ROOT / 'experiments/corpus-quick-wins-2026-10-02'
ADDITIONS = ROOT / 'experiments/local-corpus-recovery-v2'
DIVERSE = ROOT / 'experiments/joint-phrase-v19/diverse-joint'
ARMS = ('baseline', 'planner410', 'bank51')
SEEDS = (0, 1)


def read(path):
    return json.loads(Path(path).read_text())


def load(path):
    return torch.load(path, weights_only=False)


def checked(path, digest):
    if _sha(path) != digest:
        raise ValueError('artifact changed: ' + str(path))
    return path


def family_weights(families):
    counts = Counter(families)
    return np.array([1 / counts[f] for f in families], dtype=float)


def validate_additions(rows, existing, blocked):
    seen = set(existing)
    for r in rows:
        members = set(r['audio_group']['members'])
        if (r['role'] != 'train' or r['family'] in seen or r['family'] in blocked
                or r['audio_group']['blocked_by'] or members & (set(blocked) | seen)):
            raise ValueError('addition role, identity or audio-group overlap')
        seen.update(members | {r['family']})


def expanded_bank(base, additions):
    if any(d['fit_role'] != 'train' or not d['approved'] for d in additions):
        raise ValueError('bank additions must be approved train sources')
    if {d['family'] for d in additions} & {e['family'] for e in base['entries']}:
        raise ValueError('duplicate donor family')
    extra = joins.enrich_bank(workload.corrected_bank(old.retrieval_bank(additions)),
                             {d['family']: d['source'] for d in additions})
    raw = np.stack([e['descriptor'] for e in extra['entries']])
    return {**base, 'entries': base['entries'] + extra['entries'],
            'x': np.concatenate([base['x'], (raw - base['mean']) / base['sd']])}


def initialize(out, check_code=True):
    path = out / 'run.json'
    if path.exists():
        run = read(path)
        if check_code:
            checked(__file__, run['config']['code_sha256'])
        for p, h in {**run['config']['files'], **run['config']['protected']}.items():
            checked(ROOT / p, h)
        return run
    manifest = read(ADDITIONS / 'manifest.json'); verification = read(ADDITIONS / 'verification.json')
    checked(ADDITIONS / 'manifest.json', verification['manifest_sha256'])
    if manifest['status'] != 'LOCAL_CORPUS_READY' or verification['verified_sources'] != 74:
        raise ValueError('recovered corpus is not ready')
    diverse = read(DIVERSE / 'run.json')['config']
    existing = {f for f, r in diverse['sources'].items() if r['role'] == 'train'}
    blocked = set(read(ROOT / 'experiments/local-corpus-recovery-v1/run.json')['config']['blocked'])
    validate_additions(manifest['sources'], existing, blocked)
    if len(existing) != 336 or len(manifest['sources']) != 74:
        raise ValueError('unexpected training inventory')
    paths = [ADDITIONS / 'manifest.json', ADDITIONS / 'verification.json',
             budget.OUT / 'training.pt', budget.OUT / 'model.pt', budget.OUT / 'report.json',
             DIVERSE / 'run.json', approved.OUT / 'prepared.json',
             uniform.OUT / 'retrieval.pt', uniform.OUT / 'bank-receipt.json',
             old.OUT / 'run.json', old.PILOT / 'run.json', native.OUT / 'run.json',
             ROOT / 'experiments/qa-v4/comparator-v2/evaluator_freeze.json']
    prior = read(budget.OUT / 'report.json')
    checked(budget.OUT / 'training.pt', prior['training_sha256'])
    checked(budget.OUT / 'model.pt', prior['model_sha256'])
    views = read(uniform.OUT / 'bank-receipt.json')
    checked(uniform.OUT / 'retrieval.pt', views['bank_sha256'])
    for family, r in diverse['sources'].items():
        if r['role'] != 'validation':
            continue
        paths.append(checked(ROOT / r['payload'], r['payload_sha256']))
        paths.append(checked(approved.OUT / 'audio' / family.replace(':', '_') / 'features.pt',
                             diverse['validation_views'][family]['features_sha256']))
    for r in read(old.OUT / 'run.json')['config']['records']:
        if r['fit_role'] != 'development':
            continue
        key = r['family'].replace(':', '_'); data = old.OUT / 'data' / (key + '.pt')
        paths.append(checked(data, read(data.with_suffix('.json'))['sha256']))
        paths.append(checked(uniform.OUT / 'audio' / key / 'features.pt', views['audio_receipts'][r['family']]['features_sha256']))
        verify_sources(r)
        paths.append(Path(r['sources']['audio']['path']))
        fresh = ROOT / 'experiments/expressive-v1/fresh'
        paths.extend([fresh / 'mi' / key / 'gen.osu', fresh / 'partial' / key / 'b0.json'])
        for seed in SEEDS:
            folder = native.OUT / 'development' / key / 'ordered' / str(seed)
            rec = read(folder / 'record.json'); paths.append(folder / 'record.json')
            paths.extend(checked(folder / name, h) for name, h in rec['artifacts'].items())
    modules = [budget, native, old, ordered, uniform, approved, compatible, joins, workload]
    paths.extend(Path(m.__file__) for m in modules)
    paths.extend(ROOT / p for p in ['eval/joint_audio_view.py', 'eval/joint_model.py', 'eval/joint_phrase.py',
        'eval/joint_export.py', 'eval/joint_deployment.py', 'eval/joint_continuity.py', 'eval/map_reader.py',
        'eval/expression_profile.py', 'eval/quality_metrics.py', 'qa/comparator_v2.py', 'qa/comparator_support.py'])
    return freeze_run(out, {'code_sha256': _sha(__file__), 'files': {str(p): _sha(p) for p in paths},
        'protected': {p: _sha(ROOT / p) for p in PROTECTED}, 'seeds': list(SEEDS), 'arms': list(ARMS),
        'train_families': sorted(existing), 'blocked': sorted(blocked), 'ridge_penalty': .01,
        'claim': 'Exploratory component ablations; no new selector, quality gate, promotion or automatic keep/drop.'})


def source_rows(source, audio, rate):
    plans = []
    for w in source['windows']:
        counts = [0, 0, 0]
        for e in w['events']:
            hands = e['hands']; counts[2 if all(hands) else 0 if hands[0] else 1] += 1
        plans.append({'start': w['start'], 'end': w['end'], 'counts': counts})
    seconds = np.array([(p['end'] - p['start']) * 60 / source['bpm'] for p in plans])
    x = np.stack([budget.features(audio, p['start'], p['end'], source['bpm'], rate,
                                  source['duration_beats']) for p in plans])
    return x, np.array([p['counts'] for p in plans]) / (rate * seconds[:, None]), seconds, plans


def prepare(out):
    run = initialize(out); deadline = time.monotonic() + 1800
    rows = read(ADDITIONS / 'manifest.json')['sources']; base = load(budget.OUT / 'training.pt')
    if set(base['families']) != set(run['config']['train_families']) or set(base['roles']) != {'train'}:
        raise ValueError('baseline training roles changed')
    np.testing.assert_array_equal(base['weights'], family_weights(base['families']))
    xs = [base['x']]; ys = [base['y']]; families = list(base['families']); donors = []; receipts = {}
    for row in rows:
        if time.monotonic() >= deadline:
            raise TimeoutError('preparation budget exhausted; rerun same stage to resume')
        family = row['family']; verify_sources(row['record'])
        source = read(checked(row['payload'], row['payload_sha256']))
        root = out / 'audio' / family.replace(':', '_'); rp = root / 'receipt.json'
        if not rp.exists():
            render = render_common(row['record']['sources']['audio']['path'], root / 'common14800.wav')
            _torch_save(root / 'features.pt', audio_features(root / 'common14800.wav'))
            _atomic(rp, {'identity': run['identity'], 'features_sha256': _sha(root / 'features.pt'), 'render': render,
                         'source_audio_sha256': row['record']['sources']['audio']['sha256']})
            print('prepared audio', family, flush=True)
        receipt = read(rp)
        if receipt['identity'] != run['identity'] or receipt['source_audio_sha256'] != row['record']['sources']['audio']['sha256']:
            raise ValueError('audio receipt identity mismatch')
        checked(root / 'common14800.wav', receipt['render']['sha256'])
        audio = load(checked(root / 'features.pt', receipt['features_sha256']))
        rate = sum(bool(h) for e in source['events'] for h in e['hands']) / (source['duration_beats'] * 60 / source['bpm'])
        x, y, _, _ = source_rows(source, audio, rate)
        xs.append(x); ys.append(y); families.extend([family] * len(x)); receipts[family] = receipt
        is_approved = row['record']['approved_chart']
        if is_approved != any(p in row['record']['dir'] for p in APPROVED_ROOTS):
            raise ValueError('approval provenance mismatch')
        if is_approved:
            donors.append({'family': family, 'fit_role': 'train', 'approved': True, 'source': source, 'audio': audio})
    training = {'x': np.concatenate(xs), 'y': np.concatenate(ys), 'families': families,
                'weights': family_weights(families), 'roles': ['train'] * len(families)}
    if len(set(families)) != 410 or len(donors) != 6:
        raise ValueError('expanded source coverage changed')
    original = load(uniform.OUT / 'retrieval.pt'); bank = expanded_bank(original, donors)
    if len({e['family'] for e in bank['entries']}) != 51:
        raise ValueError('expected 51 donor families')
    _torch_save(out / 'training.pt', training); _torch_save(out / 'bank51.pt', bank)
    prepared = {'identity': run['identity'], 'training_sha256': _sha(out / 'training.pt'),
                'bank_sha256': _sha(out / 'bank51.pt'), 'audio_receipts': receipts,
                'train_families': 410, 'train_windows': len(families), 'bank_families': 51,
                'bank_entries': len(bank['entries']), 'old_bank_entries': len(original['entries']),
                'new_donor_families': sorted(d['family'] for d in donors)}
    _atomic(out / 'prepared.json', prepared)
    return {k: v for k, v in prepared.items() if k != 'audio_receipts'}


def fit(out):
    run = initialize(out); prepared = read(out / 'prepared.json')
    data = load(checked(out / 'training.pt', prepared['training_sha256']))
    model = budget.fit_ridge(data['x'], data['y'], data['weights'], data['roles'])
    _torch_save(out / 'planner410.pt', model)
    before = load(budget.OUT / 'model.pt'); calibration = read(approved.OUT / 'prepared.json')
    sources = read(DIVERSE / 'run.json')['config']['sources']; previous = read(budget.OUT / 'report.json')
    old_rows = {r['family']: r for r in previous['families']}; rows = []
    rate = calibration['validation_rate']; scales = calibration['validation_scales']
    for family, item in sorted(sources.items()):
        if item['role'] != 'validation':
            continue
        source = load(ROOT / item['payload'])['source']
        audio = load(approved.OUT / 'audio' / family.replace(':', '_') / 'features.pt')
        x, _, seconds, plans = source_rows(source, audio, rate); reference = budget.rhythm(plans, source['bpm'])
        arms = {}
        for name, current in [('baseline', before), ('planner410', model)]:
            predicted = budget.predict(current, x, rate, seconds, 'ridge')
            generated = budget.rhythm([{**p, 'counts': c.tolist()} for p, c in zip(plans, predicted)], source['bpm'])
            error = budget.error(reference, generated, scales)
            if name == 'baseline' and error != old_rows[family]['arms']['ridge']['errors']:
                raise ValueError('validation control did not reproduce')
            ref, got = np.array(reference), np.array(generated)
            arms[name] = {'errors': error, 'workload_ratio': float((predicted @ [1, 1, 2]).sum() / seconds.sum() / rate),
                'notes_in_reference_rest_windows': int(((ref[:, 4] == 1) & (got[:, 4] == 0)).sum()),
                'rest_over_reference_notes_windows': int(((ref[:, 4] == 0) & (got[:, 4] == 1)).sum())}
        rows.append({'family': family, 'arms': arms})
    result = {'identity': run['identity'], 'model_sha256': _sha(out / 'planner410.pt'), 'families': rows,
              'aggregate': {a: float(np.mean([r['arms'][a]['errors']['mean'] for r in rows])) for a in ('baseline', 'planner410')},
              'baseline_reproduced': True, 'decision': 'USER_REVIEW_PENDING'}
    if len(rows) != 6:
        raise ValueError('validation membership changed')
    _atomic(out / 'validation.json', result)
    return {k: v for k, v in result.items() if k != 'families'}


def development_inputs():
    result = {}
    for r in read(old.OUT / 'run.json')['config']['records']:
        if r['fit_role'] != 'development':
            continue
        family = r['family']; key = family.replace(':', '_'); fresh = ROOT / 'experiments/expressive-v1/fresh'
        source, rate, proof = deployment_input(Path(r['sources']['audio']['path']), fresh / 'mi' / key / 'gen.osu',
                                               fresh / 'partial' / key / 'b0.json')
        result[family] = {'source': source, 'rate': rate, 'proof': proof,
            'data': load(old.OUT / 'data' / (key + '.pt')),
            'audio': load(uniform.OUT / 'audio' / key / 'features.pt')}
    if len(result) != 8:
        raise ValueError('development membership changed')
    return result


def diagnostics(record, folder, item, bank, new_families):
    if not record['ok']:
        return None
    source = {**read_chart(folder / 'ExpertPlus.dat', folder / 'Info.dat'),
              'duration_beats': item['source']['duration_beats']}
    for key in ('bpm', 'walls', 'bombs'):
        if source[key] != item['source'][key]:
            raise ValueError('deployment scene changed: ' + key)
    proof = compatible.audit_ledger(source, record['donors'], bank)
    if proof['compressed'] or proof['unknown'] or proof['nonterminal_unknown_exits']:
        raise ValueError('native donor/join check failed')
    view = reference_view(source, item['data']['source'])
    np.testing.assert_allclose([b * 60 / source['bpm'] for b, *_ in source['notes']],
                               [b * 60 / view['bpm'] for b, *_ in view['notes']], rtol=0, atol=1e-10)
    ledger = record['budget_ledger']; ref = np.array(record['reference_profile']['rhythm']); got = np.array(record['profile']['rhythm'])
    gaps = [g * 60000 / source['bpm'] for h in (0, 1)
            for g in np.diff(sorted({n[0] for n in source['notes'] if n[1] == h}))]
    notes_ms = [(b * 60000 / source['bpm'], h, c, l, d) for b, h, c, l, d in source['notes']]
    pairs = {k: len(v) for k, v in lr_doubles(notes_ms).items()}
    pairs['opposite_horizontal'] = opposite_horizontal(notes_ms)
    return {'literal_donor_check': proof, 'continuity': describe(source), 'horizontal_pairs': pairs,
        'submillisecond_hand_gaps': sum(bool(g < 1) for g in gaps),
        'mean_count_error': float(np.mean([p['absolute_count_error'] for p in ledger])),
        'exact_count_windows': sum(p['absolute_count_error'] == 0 for p in ledger), 'windows': len(ledger),
        'new_donor_windows': sum(f in new_families for f, _ in record['donors']),
        'unique_donors': len(set(map(tuple, record['donors']))),
        'workload_ratio': len({(b, h) for b, h, *_ in source['notes']}) / (source['duration_beats'] * 60 / source['bpm']) / item['rate'],
        'notes_in_reference_rest_windows': int(((ref[:, 4] == 1) & (got[:, 4] == 0)).sum()),
        'rest_over_reference_notes_windows': int(((ref[:, 4] == 0) & (got[:, 4] == 1)).sum())}


def evaluate(out, cached_only=False):
    run = initialize(out, check_code=not cached_only); prepared = read(out / 'prepared.json'); validation = read(out / 'validation.json')
    banks = {'baseline': load(uniform.OUT / 'retrieval.pt'),
             'bank51': load(checked(out / 'bank51.pt', prepared['bank_sha256']))}
    models = {'baseline': load(budget.OUT / 'model.pt'),
              'planner410': load(checked(out / 'planner410.pt', validation['model_sha256']))}
    scales = read(old.PILOT / 'run.json')['config']['scales']; rows = []
    baseline_identity = read(native.OUT / 'run.json')['identity']
    deadline = time.monotonic() + 1800; kit = None
    for family, item in sorted(development_inputs().items()):
        for arm in ARMS:
            bank = banks['bank51' if arm == 'bank51' else 'baseline']
            model = models['planner410' if arm == 'planner410' else 'baseline']
            plans = native.plan(item['source'], item['audio'], model, item['rate'])
            for seed in SEEDS:
                baseline_path = native.OUT / 'development' / family.replace(':', '_') / 'ordered' / str(seed) / 'record.json'
                folder = baseline_path.parent if arm == 'baseline' else out / 'development' / family.replace(':', '_') / arm / str(seed)
                record = old.cached_record(folder / 'record.json', baseline_identity if arm == 'baseline' else run['identity'])
                if record is None:
                    if arm == 'baseline' or cached_only:
                        raise ValueError('missing authenticated cached export; review cannot generate')
                    if time.monotonic() >= deadline:
                        raise TimeoutError('evaluation budget exhausted; resume the same stage')
                    if kit is None:
                        print('loading scoped machine QA', flush=True); kit = old.machine_tools()
                    attempt = native.render(item['source'], item['audio'], bank, plans, item['rate'], seed)
                    record = ordered.measure(attempt, item['data'], folder, run['identity'], scales, kit, baseline_path)
                    print('evaluated', family, arm, seed, record['ok'], record.get('machine', {}).get('verdict'), flush=True)
                if record['plans'] != plans:
                    raise ValueError('cached plans changed')
                d = diagnostics(record, folder, item, bank, set(prepared['new_donor_families']))
                rows.append({'family': family, 'arm': arm, 'seed': seed, 'record': str(folder / 'record.json'),
                             'record_sha256': _sha(folder / 'record.json'), 'diagnostics': d})
    _atomic(out / 'evaluation.json', {'identity': run['identity'], 'rows': rows})
    initialize(out, check_code=not cached_only)
    return {'complete_records': len(rows), 'new_exports': sum(r['arm'] != 'baseline' for r in rows)}


def supported_mean(values):
    return float(np.mean(values)) if values and all(v is not None and np.isfinite(v) for v in values) else None


def report(out, reviewed=False):
    run = initialize(out, check_code=not reviewed); rows = read(out / 'evaluation.json')['rows']; arms = {}; paired = {}
    for row in rows:
        row['result'] = read(checked(row['record'], row['record_sha256']))
    if len(rows) != 48:
        raise ValueError('incomplete comparison')
    for arm in ARMS:
        all_rows = [r for r in rows if r['arm'] == arm]; good = [r for r in all_rows if r['result']['ok']]
        arms[arm] = {'complete': len(good), 'attempts': len(all_rows),
            'qa': dict(Counter(r['result'].get('machine', {}).get('verdict', 'NOT_EVALUATED') for r in all_rows)),
            'contradictions': dict(Counter(r['result'].get('machine', {}).get('contradictions', {}).get('status', 'NOT_EVALUATED') for r in all_rows)),
            'rhythm': float(np.mean([r['result']['errors']['rhythm']['mean'] for r in good])) if good else None,
            'rhythm_components': np.mean([r['result']['errors']['rhythm']['components'] for r in good], axis=0).tolist() if good else None,
            'geometry_diagnostic': float(np.mean([r['result']['errors']['geometry']['mean'] for r in good])) if good else None,
            'audio_activity_correlation': supported_mean([r['result']['profile']['audio']['activity_rms_correlation'] for r in good]),
            'new_donor_windows': sum(r['diagnostics']['new_donor_windows'] for r in good),
            'horizontal_pairs': {k: sum(r['diagnostics']['horizontal_pairs'][k] for r in good)
                                 for k in ('narrow', 'broad', 'converging', 'opposite_horizontal')},
            'max_narrow_pairs_per_map': max((r['diagnostics']['horizontal_pairs']['narrow'] for r in good), default=None),
            'continuity_excluded_groups': dict(sum((Counter(r['diagnostics']['continuity']['excluded_groups']) for r in good), Counter())),
            'mean_count_error': float(np.mean([r['diagnostics']['mean_count_error'] for r in good])) if good else None,
            'submillisecond_hand_gaps': sum(r['diagnostics']['submillisecond_hand_gaps'] for r in good),
            'fast_same_family': sum(r['diagnostics']['continuity']['bins']['fast']['same_family'] for r in good),
            'fast_opportunities': sum(r['diagnostics']['continuity']['bins']['fast']['opportunities'] for r in good),
            'medium_same_family': sum(r['diagnostics']['continuity']['bins']['medium']['same_family'] for r in good),
            'medium_opportunities': sum(r['diagnostics']['continuity']['bins']['medium']['opportunities'] for r in good),
            'notes_in_reference_rest_windows': sum(r['diagnostics']['notes_in_reference_rest_windows'] for r in good),
            'rest_over_reference_notes_windows': sum(r['diagnostics']['rest_over_reference_notes_windows'] for r in good)}
        paired[arm] = {f: float(np.mean([r['result']['errors']['rhythm']['mean'] for r in good if r['family'] == f]))
                       for f in sorted({r['family'] for r in good})}
    result = {'identity': run['identity'], 'status': 'QA_COMPLETE_USER_REVIEW_PENDING', 'arms': arms,
        'paired_rhythm': paired, 'validation': read(out / 'validation.json'),
        'prepared': {k: v for k, v in read(out / 'prepared.json').items() if k != 'audio_receipts'},
        'protected_unchanged': True, 'promotion': False,
        'generation_code_sha256': run['config']['code_sha256'], 'measurement_code_sha256': _sha(__file__),
        'measurement_dependencies': {p: _sha(ROOT / p) for p in ('motion.py', 'eval/quality_metrics.py')},
        'claim': 'Fixed-seed opened-song exploratory comparisons; scoped QA is not an enjoyment judgment. '
                 'Geometry is diagnostic only. Pros/cons and keep/drop decisions go to the user.'}
    _atomic(out / 'report.json', result)
    return {k: v for k, v in result.items() if k in ('status', 'arms', 'protected_unchanged')}


def review(out):
    if not (out / 'run.json').is_file():
        raise ValueError('review requires an existing run')
    evaluate(out, cached_only=True)
    return report(out, reviewed=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'fit', 'evaluate', 'report', 'review', 'all'])
    parser.add_argument('--out', type=Path, default=OUT)
    args = parser.parse_args(); out = args.out.resolve(); torch.set_num_threads(2)
    for stage in ('prepare', 'fit', 'evaluate', 'report') if args.stage == 'all' else (args.stage,):
        print(json.dumps(globals()[stage](out), indent=1), flush=True)


if __name__ == '__main__':
    main()
