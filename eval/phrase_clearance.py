"""Bounded clear-approach comparison; run: python -m eval.phrase_clearance.

Uses cached bank51 controls, unchanged planner/native joins, and one filtered
bank. No model fitting or production change. Keep/drop remains a user decision.
"""
import argparse
from collections import Counter
from pathlib import Path
import time

import numpy as np
import torch

from eval import corpus_quick_wins as quick
from eval.joint_phrase import ROOT, PROTECTED, _atomic, _sha
from eval.expressive_manifest import freeze_run
from eval.joint_export import read_chart
from swing_clearance import blocked_approaches

OUT = ROOT / 'experiments/phrase-clearance-full-grid-2026-10-02'


def filter_bank(bank):
    keep = [i for i, e in enumerate(bank['entries']) if not blocked_approaches(e['notes'])]
    if not keep:
        raise ValueError('clearance leaves no phrases')
    entries = [bank['entries'][i] for i in keep]
    before = {e['family'] for e in bank['entries']}; after = {e['family'] for e in entries}
    return {**bank, 'entries': entries, 'x': bank['x'][keep]}, {
        'original_entries': len(bank['entries']), 'retained_entries': len(entries),
        'removed_entries': len(bank['entries']) - len(entries),
        'retained_families': len(after), 'lost_families': sorted(before - after)}


def obstruction_counts(notes):
    hits = blocked_approaches(notes)
    return {'blocked_approaches': len(hits), 'blocked_events': len({t for t, _, _ in hits}),
            'directionless_notes': sum(n[4] == 8 for n in notes)}


def run(out):
    deadline = time.monotonic() + 1800
    parent = quick.initialize(quick.OUT, check_code=False)
    prepared = quick.read(quick.OUT / 'prepared.json')
    base = quick.load(quick.checked(quick.OUT / 'bank51.pt', prepared['bank_sha256']))
    bank, inventory = filter_bank(base)
    paths = [Path(__file__), ROOT / 'swing_clearance.py', ROOT / 'parity.py',
             Path(quick.__file__), quick.OUT / 'prepared.json', quick.OUT / 'bank51.pt']
    controls = {}
    for family in sorted(quick.development_inputs()):
        for seed in quick.SEEDS:
            path = quick.OUT / 'development' / family.replace(':', '_') / 'bank51' / str(seed) / 'record.json'
            controls[family, seed] = path
            record = quick.old.cached_record(path, parent['identity'])
            if record is None:
                raise ValueError('missing authenticated bank51 control')
            paths.append(path)
            paths.extend(path.parent / name for name in record['artifacts'])
    config = {'files': {str(p): _sha(p) for p in paths}, 'parent_identity': parent['identity'],
              'protected': {p: _sha(ROOT / p) for p in PROTECTED}, 'inventory': inventory,
              'seeds': list(quick.SEEDS), 'constraint': 'full-grid approach ray, half-grid disk; exact simultaneous cross-color pairs'}
    frozen = freeze_run(out, config)
    quick._torch_save(out / 'bank-clear.pt', bank)
    model = quick.load(quick.budget.OUT / 'model.pt')
    scales = quick.read(quick.old.PILOT / 'run.json')['config']['scales']
    rows = []; kit = None
    for family, item in sorted(quick.development_inputs().items()):
        plans = quick.native.plan(item['source'], item['audio'], model, item['rate'])
        for seed in quick.SEEDS:
            prior = controls[family, seed]
            for arm in ('bank51', 'clear'):
                folder = prior.parent if arm == 'bank51' else out / 'development' / family.replace(':', '_') / str(seed)
                identity = parent['identity'] if arm == 'bank51' else frozen['identity']
                record = quick.old.cached_record(folder / 'record.json', identity)
                if record is None:
                    if time.monotonic() >= deadline:
                        raise TimeoutError('30-minute budget exhausted; resume same command')
                    if kit is None:
                        print('loading scoped machine QA', flush=True); kit = quick.old.machine_tools()
                    attempt = quick.native.render(item['source'], item['audio'], bank, plans, item['rate'], seed)
                    if attempt['ok'] and blocked_approaches(attempt['source']['notes']):
                        raise ValueError('assembled output violates clearance')
                    record = quick.ordered.measure(attempt, item['data'], folder, identity, scales, kit, prior)
                    print(f'{family} seed {seed}: {record["ok"]} {record.get("machine", {}).get("verdict")}', flush=True)
                if record['plans'] != plans:
                    raise ValueError('cached plans changed')
                diag = quick.diagnostics(record, folder, item, base if arm == 'bank51' else bank,
                                         set(prepared['new_donor_families']))
                if record['ok']:
                    diag.update(obstruction_counts(read_chart(folder / 'ExpertPlus.dat', folder / 'Info.dat')['notes']))
                    if arm == 'clear' and diag['blocked_approaches']:
                        raise ValueError('export violates clearance')
                rows.append({'family': family, 'seed': seed, 'arm': arm, 'record': str(folder / 'record.json'),
                             'record_sha256': _sha(folder / 'record.json'), 'diagnostics': diag, 'result': record})
    arms = {}
    for arm in ('bank51', 'clear'):
        attempts = [r for r in rows if r['arm'] == arm]; good = [r for r in attempts if r['result']['ok']]
        arms[arm] = {'complete': len(good), 'attempts': len(attempts),
            'qa': dict(Counter(r['result'].get('machine', {}).get('verdict', 'NOT_EVALUATED') for r in attempts)),
            'contradictions': dict(Counter(r['result'].get('machine', {}).get('contradictions', {}).get('status', 'NOT_EVALUATED') for r in attempts)),
            'rhythm': float(np.mean([r['result']['errors']['rhythm']['mean'] for r in good])) if good else None,
            'mean_count_error': float(np.mean([r['diagnostics']['mean_count_error'] for r in good])) if good else None,
            'audio_activity_correlation': quick.supported_mean([r['result']['profile']['audio']['activity_rms_correlation'] for r in good]),
            'horizontal_pairs': {k: sum(r['diagnostics']['horizontal_pairs'][k] for r in good)
                                 for k in ('narrow', 'broad', 'converging', 'opposite_horizontal')},
            'counts': {k: sum(r['diagnostics'][k] for r in good) for k in ('blocked_approaches', 'blocked_events',
                'directionless_notes', 'exact_count_windows', 'windows', 'submillisecond_hand_gaps',
                'notes_in_reference_rest_windows', 'rest_over_reference_notes_windows')},
            'continuity': {b: {k: sum(r['diagnostics']['continuity']['bins'][b][k] for r in good)
                              for k in ('same_family', 'opportunities')} for b in ('fast', 'medium')},
            'excluded_groups': dict(sum((Counter(r['diagnostics']['continuity']['excluded_groups']) for r in good), Counter()))}
    quick.initialize(quick.OUT, check_code=False)
    for p, digest in {**config['files'], **config['protected']}.items():
        quick.checked(ROOT / p, digest)
    report = {'identity': frozen['identity'], 'status': 'QA_COMPLETE_USER_REVIEW_PENDING',
              'inventory': inventory, 'bank_sha256': _sha(out / 'bank-clear.pt'), 'arms': arms,
              'protected_unchanged': True, 'promotion': False,
              'paired_rhythm': {arm: {f: quick.supported_mean([r['result']['errors']['rhythm']['mean']
                  for r in rows if r['arm'] == arm and r['family'] == f and r['result']['ok']])
                  for f in sorted({r['family'] for r in rows})} for arm in arms}}
    _atomic(out / 'evaluation.json', {'identity': frozen['identity'], 'rows': rows})
    _atomic(out / 'report.json', report)
    return report


if __name__ == '__main__':
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=OUT)
    args = parser.parse_args(); torch.set_num_threads(2)
    print(json.dumps(run(args.out.resolve()), indent=1), flush=True)
