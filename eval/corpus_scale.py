"""Isolated human-map corpus scaling for the production Flow architecture.

python -m eval.corpus_scale acquire --target 2000
All receipts and maps are resumable and separate from the shipped corpus.
"""
import argparse
from collections import Counter
from contextlib import redirect_stdout
import hashlib
import io
import json
import random
import subprocess
from pathlib import Path
import shutil
import time
import zipfile

import requests
import numpy as np
import torch

import groom
from eval.corpus import _norm
from eval.joint_phrase import ROOT, PROTECTED, _sha, _atomic

OUT = ROOT / 'experiments/corpus-scale-2026-10-02'
BASE = ROOT / 'experiments/clean-flow-warm-v1'
API = 'https://api.beatsaver.com'
QUERY = dict(q='', order='Latest', minRating=.85, minVotes=50,
             minDuration=60, maxDuration=600, noodle='false', me='false', pageSize=100)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def title_key(title, artist):
    return _norm(title) + '|' + _norm(artist)


def eligible(doc):
    stats = doc.get('stats', {}); versions = doc.get('versions', [])
    return (doc.get('automapper') is False and bool(versions)
        and versions[0].get('state') == 'Published'
        and stats.get('score', 0) >= .85
        and stats.get('upvotes', 0) + stats.get('downvotes', 0) >= 50
        and 60 <= doc.get('metadata', {}).get('duration', 0) <= 600
        and any(d.get('characteristic') == 'Standard' and not d.get('me')
                and not d.get('ne') and d.get('notes', 0) > groom.CTX
                for d in versions[0].get('diffs', [])))


def unpack(data, dest):
    """Read only chart/audio members, bounded, without archive path traversal."""
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        members = [m for m in archive.infolist() if not m.is_dir()
                   and Path(m.filename).suffix.lower() in ('.dat', '.egg', '.ogg', '.wav', '.mp3')]
        names = [Path(m.filename).name for m in members]
        if (len({n.lower() for n in names}) != len(names)
                or sum(m.file_size for m in members) > 128 * 1024**2
                or not any(n.lower() == 'info.dat' for n in names)):
            raise ValueError('unsupported or oversized archive')
        dest.mkdir(parents=True, exist_ok=True)
        for member, name in zip(members, names):
            (dest / name).write_bytes(archive.read(member))


class Client:
    def __init__(self):
        self.last = 0.
        self.session = requests.Session()
        self.session.headers['User-Agent'] = 'osu2bs-research/1.0'

    def get(self, url, **kwargs):
        for attempt in range(3):
            time.sleep(max(0., self.last + 1. - time.monotonic()))
            self.last = time.monotonic()
            try:
                response = self.session.get(url, timeout=(15, 45), **kwargs)
                if response.status_code in (429, 502, 503, 504):
                    time.sleep(min(45., float(response.headers.get('Retry-After', 5 * (attempt+1)))))
                    continue
                response.raise_for_status()
                return response
            except requests.RequestException:
                if attempt == 2:
                    raise
        raise RuntimeError('service unavailable after three attempts')


def baseline(out):
    """Authenticate old prepared data without rebuilding old experiment chains."""
    path = out / 'baseline.json'
    if path.exists():
        info = read(path)
        for p, digest in info['inputs'].items():
            if _sha(ROOT / p) != digest:
                raise ValueError('baseline input changed: ' + p)
        return info
    from eval.prepared_dataset import read_prepared
    identity = read(BASE / 'dataset.identity.json')['identity']
    train, val, _ = read_prepared(BASE, identity)
    info = {'train_families': len({m[6] for m in train}), 'train_charts': len(train),
            'val_families': len({m[6] for m in val}), 'val_charts': len(val),
            'inputs': {str(p): _sha(p) for p in (BASE / 'dataset.pt', BASE / 'dataset.identity.json',
                ROOT / 'eval/corpus_manifest.json', ROOT / 'eval/clean_rhythm_panel.json')},
            'protected': {p: _sha(ROOT / p) for p in PROTECTED}, 'query': QUERY}
    if (info['train_families'], info['val_families']) != (747, 63):
        raise ValueError('unexpected baseline inventory')
    _atomic(path, info)
    return info


def inventory(out):
    path = out / 'existing.json'
    if path.exists():
        return read(path)
    maps = read(ROOT / 'eval/corpus_manifest.json')['maps']
    seen = {'ids': set(), 'titles': set(), 'charts': set(), 'audio': set()}
    for m in maps:
        seen['ids'].add(m['map_id']); seen['titles'].add(m['title_key'])
        seen['charts'].update(c['sha256'] for c in m['charts'].values())
        if m.get('audio_file'):
            path = Path(m['dir']) / m['audio_file']
            if path.exists():
                seen['audio'].add(_sha(path))
    # Includes all existing splits, not just the training subset.
    result = {k: sorted(v) for k, v in seen.items()}
    _atomic(out / 'existing.json', result)
    return result


def acquire(out, target):
    baseline(out); seen = {k: set(v) for k, v in inventory(out).items()}
    out.mkdir(parents=True, exist_ok=True)
    if not (out / 'acquire-source.py').exists():
        (out / 'acquire-source.py').write_bytes(Path(__file__).read_bytes())
    receipt_paths = sorted((out / 'receipts').glob('*.json'))
    accepted = []; attempted = set(); exclusions = Counter()
    for path in receipt_paths:
        r = read(path); attempted.add(r['id']); exclusions[r['status']] += 1
        if r['status'] == 'accepted':
            for p, digest in r['files'].items():
                if _sha(out / p) != digest:
                    raise ValueError('downloaded source changed: ' + p)
            accepted.append(r)
            for k, vals in r['identity'].items():
                seen[k].update(vals)
    progress = read(out / 'progress.json') if (out / 'progress.json').exists() else dict(page=0, before=None)
    client = Client(); consecutive_errors = 0
    while len(accepted) < target and len(attempted) < 6000:
        page = progress['page']; before = progress['before']
        page_path = out / 'pages' / f'{before or "initial"}-{page}.json'
        if not page_path.exists():
            query = {**QUERY, **({'to': before} if before else {})}
            payload = client.get(API + f'/search/text/{page}', params=query).json()
            _atomic(page_path, payload)
        docs = read(page_path).get('docs', [])
        if not docs:
            raise RuntimeError('eligible catalog exhausted before target')
        for doc in docs:
            mid = doc['id']
            if mid in attempted or mid in seen['ids'] or not eligible(doc):
                continue
            key = title_key(doc['metadata']['songName'], doc['metadata']['songAuthorName'])
            if key in seen['titles']:
                continue
            row = {'id': mid, 'metadata': doc, 'status': 'pending'}
            dest = out / 'maps' / mid
            try:
                if not dest.exists():
                    url = doc['versions'][0]['downloadURL']
                    response = client.get(url, stream=True)
                    chunks = []; size = 0
                    with response:
                        for chunk in response.iter_content(1024**2):
                            size += len(chunk)
                            if size > 48 * 1024**2:
                                raise ValueError('archive exceeds 48 MiB')
                            chunks.append(chunk)
                    stage = out / 'staging' / mid
                    if stage.exists(): shutil.rmtree(stage)
                    unpack(b''.join(chunks), stage)
                    dest.parent.mkdir(parents=True, exist_ok=True); stage.rename(dest)
                ip = next(p for p in dest.iterdir() if p.name.lower() == 'info.dat')
                info = read(ip)
                actual = title_key(info.get('_songName', info.get('song', {}).get('title', '')),
                                   info.get('_songAuthorName', info.get('song', {}).get('author', '')))
                files = {str(p.relative_to(out)): _sha(p) for p in dest.iterdir() if p.is_file()}
                charts = {_sha(p) for p in dest.iterdir() if p.suffix.lower() == '.dat' and p != ip}
                audio = {_sha(p) for p in dest.iterdir() if p.suffix.lower() in ('.ogg', '.egg', '.mp3', '.wav')}
                if (actual in seen['titles'] or charts & seen['charts'] or audio & seen['audio']):
                    raise ValueError('existing song/chart/audio identity')
                # Cheap loader eligibility only; zeros NEVER enter training.
                with redirect_stdout(io.StringIO()):
                    samples = groom.load_map_all(dest, feature_provider=lambda p, times: torch.zeros(len(times), groom.N_AUDIO))
                usable = {name: m for name, m in samples.items() if len(m[2]) > groom.CTX}
                if not usable:
                    raise ValueError('no supported >256-event training chart')
                ident = dict(ids=[mid], titles=sorted({key, actual}), charts=sorted(charts), audio=sorted(audio))
                row.update(status='accepted', files=files, identity=ident,
                           usable_charts={name: len(m[2]) for name, m in usable.items()},
                           ordinal=len(accepted))
                accepted.append(row)
                for k, vals in ident.items(): seen[k].update(vals)
                consecutive_errors = 0
            except requests.RequestException:
                consecutive_errors += 1
                if consecutive_errors >= 3: raise
                continue
            except (ValueError, zipfile.BadZipFile, StopIteration, KeyError) as e:
                row.update(status='excluded', reason=str(e))
            attempted.add(mid); exclusions[row['status']] += 1
            _atomic(out / 'receipts' / (mid + '.json'), row)
            if len(accepted) % 25 == 0:
                print(f'acquired {len(accepted)}/{target} usable families; {len(attempted)} downloads screened', flush=True)
            _atomic(out / 'manifest.json', {'status': 'ACQUIRING', 'baseline_families': 747,
                'target': target, 'families': sorted(accepted, key=lambda r: r['ordinal']),
                'counts': dict(exclusions)})
            if len(accepted) >= target: break
        progress['page'] += 1
        # Avoid search pagination limits; advance date cursor after 500 results.
        if progress['page'] == 5:
            progress = dict(page=0, before=docs[-1]['uploaded'])
        _atomic(out / 'progress.json', progress)
    if len(accepted) < target:
        raise RuntimeError('6000-download budget exhausted before target')
    manifest = read(out / 'manifest.json'); manifest['status'] = 'ACQUIRED_PENDING_AUDIO_SCREEN_AND_FEATURES'
    _atomic(out / 'manifest.json', manifest)
    return {'accepted': len(accepted), 'target': target, 'status': manifest['status']}


def fit(out, addition):
    if addition < 0: raise ValueError('fit requires a nonnegative corpus addition')
    base = baseline(out)
    from eval.prepared_dataset import read_prepared
    train, val, _ = read_prepared(BASE, read(BASE / 'dataset.identity.json')['identity'])
    if addition:
        manifest = read(out / 'prepared.json')
        if manifest['status'] != 'READY' or len(manifest['families']) < addition:
            raise ValueError('not enough screened/prepared additional families')
        for row in manifest['families'][:addition]:
            path = out / row['tensor']
            if _sha(path) != row['sha256']: raise ValueError('changed prepared addition')
            train.extend(torch.load(path, weights_only=False))
    if len({m[6] for m in train}) != 747 + addition:
        raise ValueError('training family count mismatch')
    dest = out / f'fit-{addition}'; dest.mkdir(exist_ok=True)
    if (dest / 'result.json').exists():
        result = read(dest / 'result.json')
        if _sha(dest / 'flow.pt') != result['checkpoint_sha256']: raise ValueError('checkpoint changed')
        return result
    torch.manual_seed(20261002); torch.set_num_threads(2)
    config = dict(additional_families=addition, train_families=747+addition,
        effective_flow_families=len({m[6] for m in train if len(m[2])>groom.CTX}),
        train_charts=len(train), validation_families=63, seed=20261002,
        max_updates=18000, architecture='Flow', device=torch.cuda.get_device_name(0),
        code={p: _sha(ROOT/p) for p in ('groom.py', 'eval/corpus_scale.py')},
        baseline_inputs=base['inputs'], prepared_sha256=_sha(out/'prepared.json') if addition else None)
    _atomic(dest/'config.json', config)
    started = time.monotonic()
    history = groom.train_flow(train, val, 'cuda', random.Random(20261002),
        out_path=dest/'flow.pt', history_path=dest/'history.json', max_epochs=600, steps_per_epoch=30)
    model = groom.Flow().cuda(); model.load_state_dict(torch.load(dest/'flow.pt', weights_only=False))
    loss, accuracy = groom.eval_flow(model, val, 'cuda')
    if abs(loss-history['best_val_loss']) > 1e-5: raise ValueError('reload validation mismatch')
    for p, digest in base['protected'].items():
        if _sha(ROOT/p) != digest: raise ValueError('protected artifact changed')
    result = dict(**config, checkpoint_sha256=_sha(dest/'flow.pt'), validation_loss=loss,
        direction_accuracy=accuracy, updates=history['updates'], elapsed_s=time.monotonic()-started,
        status='TRAINED_PENDING_GENERATED_MAP_QA')
    _atomic(dest/'result.json', result)
    return result


def envelope(path):
    """Cheap original-audio envelope for cross-split near-copy screening."""
    raw = subprocess.run(['ffmpeg', '-v', 'error', '-i', str(path), '-t', '90',
        '-ac', '1', '-ar', '4000', '-f', 'f32le', '-'], check=True, capture_output=True).stdout
    y = np.frombuffer(raw, dtype='<f4').astype(float)
    # 50 Hz non-overlapping RMS; lagged Pearson comparison is existing code.
    y = y[:len(y)//80*80].reshape(-1, 80)
    return np.sqrt(np.mean(y*y, axis=1))


def heldout_audio(out):
    path = out / 'heldout-audio.pt'
    if path.exists(): return torch.load(path, weights_only=False)
    rows = []; seen = set()
    for m in read(ROOT/'eval/corpus_manifest.json')['maps']:
        if m['split'] == 'train' or not m.get('audio_file'): continue
        ap = Path(m['dir'])/m['audio_file']
        digest = _sha(ap)
        if digest in seen: continue
        seen.add(digest)
        rows.append(dict(family=m['family'], path=str(ap), sha256=digest, envelope=envelope(ap)))
    torch.save(rows, path)
    print('prepared held-out audio screen', len(rows), flush=True)
    return rows


def prepare(out, follow=False, workers=4):
    from eval.map_load_pool import OrderedMapPool
    from eval.joint_audio_identity import peak_correlation
    base = baseline(out)
    heldout = heldout_audio(out)
    for r in heldout:
        if _sha(r['path']) != r['sha256']: raise ValueError('held-out audio changed')
    (out/'tensors').mkdir(exist_ok=True)
    cache = out/'features.pt'
    if not cache.exists(): shutil.copy2(BASE/'feats_cache.pt', cache)
    with OrderedMapPool(workers, cache) as pool:
        while True:
            manifest = read(out/'manifest.json')
            todo = [r for r in sorted(manifest['families'], key=lambda r:r['ordinal'])
                    if not (out/'tensors'/(r['id']+'.json')).exists()][:25]
            if not todo:
                if manifest['status'] == 'ACQUIRING' and follow:
                    time.sleep(20); continue
                break
            screened = []
            for row in todo:
                dest = out/'maps'/row['id']
                for p,h in row['files'].items():
                    if _sha(out/p)!=h: raise ValueError('source changed before preparation')
                ip=next(p for p in dest.iterdir() if p.name.lower()=='info.dat'); info=read(ip)
                audio_name=info.get('_songFilename') or info.get('audio',{}).get('songFilename')
                ap=dest/audio_name if audio_name and (dest/audio_name).is_file() else next(
                    p for p in dest.iterdir() if p.suffix.lower() in ('.ogg','.egg'))
                env=envelope(ap); overlap=None
                for h in heldout:
                    score=peak_correlation(env,h['envelope'])['correlation']
                    if score is not None and score>=.95:
                        overlap=dict(family=h['family'],correlation=score);break
                if overlap:
                    _atomic(out/'tensors'/(row['id']+'.json'),dict(id=row['id'],status='AUDIO_OVERLAP',match=overlap))
                else: screened.append(row)
            samples = pool.map([out/'maps'/r['id'] for r in screened])
            for row, charts in zip(screened,samples):
                usable=[m for m in charts.values() if len(m[2])>groom.CTX]
                rec=dict(id=row['id'],status='FEATURE_FAILED',ordinal=row['ordinal'])
                if usable:
                    rows=[m+(1./len(usable),'new:'+row['id']) for m in usable]
                    path=out/'tensors'/(row['id']+'.pt');torch.save(rows,path)
                    rec.update(status='READY',tensor=str(path.relative_to(out)),sha256=_sha(path),charts=len(rows))
                _atomic(out/'tensors'/(row['id']+'.json'),rec)
            ready=[read(p) for p in (out/'tensors').glob('*.json')]
            print('prepared families',sum(r['status']=='READY' for r in ready),'screened',len(ready),flush=True)
    prepared=[read(p) for p in (out/'tensors').glob('*.json')]
    ready=sorted([r for r in prepared if r['status']=='READY'],
                 key=lambda r:hashlib.sha256(('corpus-scale-20261002:'+r['id']).encode()).hexdigest())
    result=dict(status='READY' if len(ready)>=2000 else 'SHORTFALL',families=ready[:2000],
                available_families=len(ready),exclusions=dict(Counter(r['status'] for r in prepared)),
                feature_stats=pool.stats,baseline_dataset_families=base['train_families'],
                baseline_effective_flow_families=739,heldout_audio_sha256=_sha(out/'heldout-audio.pt'))
    _atomic(out/'prepared.json',result)
    return {k:v for k,v in result.items() if k not in ('families','feature_stats')}


def evaluate(out, addition, policy, *, model_kwargs=None):
    """Paired fixed schedules, all seeds reported; no candidate replacement."""
    import convert
    import motion
    from eval import joint_pilot as qa
    from eval.joint_continuity import describe
    from eval.joint_export import export_chart
    from eval.quality_metrics import immutable_signature, four_gram_stats
    from flow_decode import EventSchedule
    from swing_clearance import blocked_approaches
    baseline(out)
    checkpoint = ROOT/'flow.pt' if addition == -1 else out/f'fit-{addition}'/'flow.pt'
    model = groom.Flow(**(model_kwargs or {}))
    model.load_state_dict(torch.load(checkpoint, weights_only=False)); model.eval()
    panel = read(ROOT/'eval/clean_rhythm_panel.json')['entries']
    if len(panel)!=8: raise ValueError('panel membership changed')
    dest=out/'evaluation'/f'{addition}-{policy}'; dest.mkdir(parents=True,exist_ok=True)
    config=dict(checkpoint_sha256=_sha(checkpoint), policy=policy, seeds=[0,1],
                code={p:_sha(ROOT/p) for p in ('groom.py','learned_geometry.py','swing_clearance.py','eval/corpus_scale.py')})
    if model_kwargs is not None:
        config.update(architecture=model_kwargs, timing_source_sha256=_sha(ROOT/'timing.py'))
    if (dest/'config.json').exists() and read(dest/'config.json')!=config:
        raise ValueError('evaluation code/checkpoint changed; use a fresh output directory')
    _atomic(dest/'config.json',config)
    kit=None; rows=[]
    for e in panel:
        for key in ('osu','audio'):
            if _sha(ROOT/e[key])!=e[key+'_sha256']: raise ValueError('panel source changed')
        meta,objects,bpm,offset=convert.parse_osu(ROOT/e['osu'])
        steps,T,step_ms,offset,grid=convert.grid_steps(objects,bpm,offset,meta.get('_timing'))
        # Isolated inference feature cache, not the shipped cache or preparation writer.
        prior_cache=groom._FEAT_CACHE;groom._FEAT_CACHE=out/'evaluation-features.pt';groom._FEATS.clear()
        try: audio=groom.cached_audio_features(ROOT/e['audio'],[grid.time(s) for s in range(T)])
        finally:groom._FEAT_CACHE=prior_cache;groom._FEATS.clear()
        spec=convert.diff_spec('ExpertPlus')
        for seed in (0,1):
            rp=dest/e['song']/str(seed)/'record.json'
            if rp.exists():
                r=read(rp)
                for p,h in r.get('artifacts',{}).items():
                    if _sha(rp.parent/p)!=h: raise ValueError('cached output changed')
                rows.append(r);continue
            control=out/'schedules'/e['song']/f'{seed}.pt'
            if control.exists(): raw0,walls0=torch.load(control,weights_only=False)
            else:
                raw0,walls0=groom.groom_notes(steps,T,step_ms,offset,afeat=audio,seed=seed,grid=grid,
                    replay_mode='off',drate=spec['cond'],band_scale=spec['scale'],band=spec['band'])
                control.parent.mkdir(parents=True,exist_ok=True);torch.save((raw0,walls0),control)
            schedule=EventSchedule.from_raw(raw0,walls0,grid,T,step_ms)
            trace={};started=time.monotonic()
            r=dict(song=e['song'],seed=seed,policy=policy,addition=addition,schedule_sha256=_sha(control))
            try:
                raw,_=groom.groom_notes(steps,T,step_ms,offset,afeat=audio,flow=model,seed=seed,grid=grid,
                    replay_mode='off',drate=spec['cond'],band_scale=spec['scale'],band=spec['band'],
                    schedule_in=schedule.entries,schedule_mask=True,geometry_policy=policy,trace=trace)
                r['decode_elapsed_s']=time.monotonic()-started
                if immutable_signature(raw,walls0,grid)!=immutable_signature(raw0,walls0,grid):
                    raise ValueError('fixed schedule not honored')
                if trace.get('schedule_infeasible'): raise ValueError('pinned chain infeasible')
                if blocked_approaches(raw): raise ValueError('clearance rule violated')
                if policy=='learned' and trace['cut_clearance']['events_changed']:
                    raise ValueError('learned policy required posthoc clearance repair')
                ns=[dict(t=grid.time(s),hand=h,col=c,layer=l,dir=d) for s,h,c,l,d in raw]
                ws=[dict(t=grid.time(s),dur=grid.time(s+n)-grid.time(s),col=c) for s,n,c in walls0]
                ms=[(n['t'],n['hand'],n['col'],n['layer'],n['dir']) for n in ns]
                source=dict(bpm=bpm,duration_beats=grid.time(T)*bpm/60000,
                    notes=[(t*bpm/60000,h,c,l,d) for t,h,c,l,d in ms],bombs=[],
                    walls=[(w['t']*bpm/60000,w['dur']*bpm/60000,w['col'],1,0,5) for w in ws])
                emitted=export_chart(source,rp.parent)
                if blocked_approaches(emitted['notes']): raise ValueError('export clearance violated')
                if kit is None: print('loading scoped machine QA',flush=True);kit=qa.machine_tools()
                r.update(ok=True,problems=convert.check(ns,bpm,ws,convert.NPS_CAP*spec['scale']),
                    machine=qa.machine_measure(emitted,kit),motion=motion.report(ms),
                    continuity=describe(source),repetition=four_gram_stats(ms,60000/bpm),
                    notes=len(raw),clearance=trace['cut_clearance'],
                    artifacts={p:_sha(rp.parent/p) for p in ('ExpertPlus.dat','Info.dat')})
            except ValueError as exc:
                # Rhythm traces contain tensors; preserve relevant diagnostics
                # without making a failed candidate impossible to serialize.
                r.update(ok=False,reason=str(exc),trace={k:trace[k] for k in
                    ('schedule','schedule_infeasible','cut_clearance') if k in trace})
            r['elapsed_s']=time.monotonic()-started
            _atomic(rp,r);rows.append(r)
            print('evaluated',addition,policy,e['song'],seed,r['ok'],r.get('reason',r.get('problems')),flush=True)
    good=[r for r in rows if r['ok']]
    summary=dict(config=config,attempts=len(rows),complete=len(good),
        qa=dict(Counter(r.get('machine',{}).get('verdict','NOT_EVALUATED') for r in rows)),
        valid=sum(not r['problems'] for r in good),
        mean_flags_per_1000=float(np.mean([r['motion']['flags']['flags_per_1000'] for r in good])) if good else None,
        clearance_repairs=sum(r['clearance']['events_changed'] for r in good),
        families=rows,status='QA_COMPLETE_USER_REVIEW_PENDING')
    _atomic(dest/'report.json',summary)
    return {k:v for k,v in summary.items() if k not in ('families','config')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['acquire', 'prepare', 'fit', 'evaluate'])
    parser.add_argument('--out', type=Path, default=OUT)
    parser.add_argument('--target', type=int, default=2000)
    parser.add_argument('--addition', type=int, choices=[-1, 0, 500, 1000, 2000], default=0)
    parser.add_argument('--policy', choices=['legacy','learned'], default='legacy')
    parser.add_argument('--follow', action='store_true')
    parser.add_argument('--workers', type=int, choices=range(1,9), default=4)
    args = parser.parse_args(); torch.set_num_threads(2)
    if args.stage == 'acquire': result = acquire(args.out.resolve(), args.target)
    elif args.stage == 'prepare': result = prepare(args.out.resolve(), args.follow, args.workers)
    elif args.stage == 'evaluate': result = evaluate(args.out.resolve(), args.addition, args.policy)
    else: result = fit(args.out.resolve(), args.addition)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
