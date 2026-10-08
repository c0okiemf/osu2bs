"""Three bounded Flow fits on the frozen +500 human-map corpus.

python -m eval.flow_training_returns prepare|fit|evaluate --arm budget|capacity|timing
Uses existing corpus/evaluation code; never writes production checkpoints.
"""
import argparse
from collections import defaultdict
import json
import random
import shutil
import time
from pathlib import Path

import torch
import groom
from eval import corpus_scale as corpus
from eval.joint_phrase import ROOT, _sha, _atomic

OLD = corpus.OUT
OUT = ROOT/'experiments/flow-training-returns-2026-10-02'
ARMS = {'budget':dict(width=192,timing=False),
        'capacity':dict(width=256,timing=False),
        'timing':dict(width=192,timing=True)}
SCHEMA = 'elapsed-ms-log1p-250-plus-local-beat-ms-and-history-flags-v1'
CODE = ('groom.py','timing.py','learned_geometry.py','eval/flow_training_returns.py','eval/corpus_scale.py')


def source_data(addition=500, pool_dir=OLD):
    base=corpus.baseline(pool_dir)
    for p,h in base['protected'].items():
        if _sha(ROOT/p)!=h:raise ValueError('production artifact changed: '+p)
    from eval.prepared_dataset import read_prepared
    train,val,_=read_prepared(corpus.BASE,corpus.read(corpus.BASE/'dataset.identity.json')['identity'])
    pool=corpus.read(pool_dir/'prepared.json')
    if addition < 0 or addition > len(pool['families']):
        raise ValueError('requested corpus prefix unavailable')
    for row in pool['families'][:addition]:
        p=pool_dir/row['tensor']
        if _sha(p)!=row['sha256']:raise ValueError('addition tensor changed')
        train.extend(torch.load(p,weights_only=False))
    assert len({m[6] for m in train})==747+addition
    return train,val,base


def prepare(out=OUT, addition=500, pool_dir=OLD, clock_cache=None):
    out.mkdir(parents=True,exist_ok=True)
    train,val,base=source_data(addition, pool_dir)
    dest=out/'prepared.json'
    identity=dict(addition=addition,inputs=base['inputs'],pool_sha256=_sha(pool_dir/'prepared.json'),
                  dataset_schema=SCHEMA,
                  code={p:_sha(ROOT/p) for p in ('groom.py','timing.py')})
    if pool_dir != OLD:
        identity['pool_dir']=str(pool_dir.resolve())
    if clock_cache is not None:
        identity['clock_cache']=dict(path=str(clock_cache.resolve()),
            prepared_sha256=_sha(clock_cache/'prepared.json'),grids_sha256=_sha(clock_cache/'grids.pt'))
    if dest.exists():
        result=corpus.read(dest)
        if result['identity']!=identity or _sha(out/'grids.pt')!=result['grids_sha256']:
            raise ValueError('prepared timing identity changed')
        for p,h in result['sources'].items():
            if _sha(Path(p))!=h:raise ValueError('timing source changed: '+p)
        return result
    sources={};ledger=[];grids={};groups=defaultdict(list)
    cached={};cached_ledger=defaultdict(list)
    if clock_cache is not None:
        old=corpus.read(clock_cache/'prepared.json')
        if old['identity']['code']!=identity['code'] or old['identity']['inputs']!=identity['inputs']:
            raise ValueError('clock cache parser/base identity changed')
        if old['grids_sha256']!=identity['clock_cache']['grids_sha256']:
            raise ValueError('clock cache changed')
        old_pool_dir=Path(old['identity'].get('pool_dir',str(OLD)))
        if _sha(old_pool_dir/'prepared.json')!=old['identity']['pool_sha256']:
            raise ValueError('clock cache corpus changed')
        n=old['identity']['addition']
        if n>addition or corpus.read(pool_dir/'prepared.json')['families'][:n]!=corpus.read(old_pool_dir/'prepared.json')['families'][:n]:
            raise ValueError('clock cache tensor prefix changed')
        trgrid,vagrid=torch.load(clock_cache/'grids.pt',weights_only=False)
        if len(trgrid)!=old['train_charts'] or len(trgrid)>len(train) or len(vagrid)!=len(val):
            raise ValueError('clock cache row count changed')
        # The authenticated tensor prefix and unchanged base determine row order.
        cached={id(m):g for m,g in zip(train[:len(trgrid)]+val,trgrid+vagrid)}
        for p,h in old['sources'].items():
            if _sha(Path(p))!=h:raise ValueError('cached timing source changed: '+p)
        sources.update(old['sources'])
        for row in old['ledger']:cached_ledger[row['family']].append(row)
    for m in train+val:groups[m[6]].append(m)
    existing={r['dir']:r for r in corpus.read(ROOT/'eval/corpus_manifest.json')['maps']}
    acquired={r['id']:r for r in corpus.read(pool_dir/'manifest.json')['families']}
    for index,(family,maps) in enumerate(groups.items()):
        if all(id(m) in cached for m in maps):
            for m in maps:grids[id(m)]=cached[id(m)]
            ledger.extend(cached_ledger[family])
            continue
        if family.startswith('new:'):
            row=acquired[family[4:]];directory=pool_dir/'maps'/row['id']
            expected={str(pool_dir/p):h for p,h in row['files'].items() if p.lower().endswith('.dat')}
        else:
            directory=Path(family);row=existing[str(directory)]
            expected={str(directory/r['file']):r['sha256'] for r in row['charts'].values()}
        ip=next(p for p in directory.iterdir() if p.name.lower()=='info.dat')
        info=corpus.read(ip);bpm=info.get('_beatsPerMinute') or info.get('audio',{}).get('bpm')
        if bpm is None or float(bpm)<=0:raise ValueError('missing authoritative BPM')
        if not family.startswith('new:') and float(bpm)!=float(row['bpm']):
            raise ValueError('manifest/source BPM mismatch')
        for p,h in expected.items():
            if _sha(Path(p))!=h:raise ValueError('changed chart: '+p)
            sources[p]=h
        sources[str(ip)]=_sha(ip)
        # Parse only to bind each cached chart to its authoritative clock.
        # Dummy audio is discarded; all training inputs remain the saved tensors.
        loaded=groom.load_map_all(directory,with_timing=True,
            feature_provider=lambda _p,ts:torch.zeros(len(ts),4))
        for m in maps:
            matches=[(name,v) for name,v in loaded.items() if v[2]==m[2]
                     and torch.equal(v[1],m[1]) and torch.equal(v[3],m[3]) and torch.equal(v[4],m[4])]
            if not matches:raise ValueError('cannot bind cached chart clock: '+family)
            grid=matches[0][1][5]
            if any(v[5].times!=grid.times for _,v in matches):
                raise ValueError('ambiguous identical notes with different clocks')
            grids[id(m)]=grid
            uniform=60000/float(bpm)/4
            deviation=max(abs(grid.gap_ms(0,e[0])-e[0]*uniform) for e in m[2])
            ledger.append(dict(family=family,difficulties=[n for n,_ in matches],
                bpm=bpm,events=len(m[2]),max_elapsed_difference_from_old_uniform_ms=deviation))
        if (index+1)%100==0:print('bound source clocks',index+1,'/',len(groups),flush=True)
    ordered=([grids[id(m)] for m in train],
             [grids[id(m)] for m in val])
    torch.save(ordered,out/'grids.pt')
    _atomic(out/'baseline.json',base)
    shutil.copytree(OLD/'schedules',out/'schedules',dirs_exist_ok=True)
    shutil.copyfile(OLD/'evaluation-features.pt',out/'evaluation-features.pt')
    result=dict(identity=identity,grids_sha256=_sha(out/'grids.pt'),sources=sources,
                train_charts=len(train),val_charts=len(val),ledger=ledger,reused_clock_rows=len(cached),
                windows=dict(first='0',middle='floor((N-256)/2)',end='N-256'),
                status='PREPARED',source_format='https://bsmg.wiki/mapping/map-format/beatmap.html#bpm-events')
    _atomic(dest,result)
    return result


def data(timing):
    train,val,base=source_data()
    prepared=corpus.read(OUT/'prepared.json')
    if prepared['identity']['pool_sha256']!=_sha(OLD/'prepared.json'):
        raise ValueError('frozen pool changed')
    if timing:
        if _sha(OUT/'grids.pt')!=prepared['grids_sha256']:raise ValueError('clock cache changed')
        trgrid,vagrid=torch.load(OUT/'grids.pt',weights_only=False)
        assert len(trgrid)==len(train) and len(vagrid)==len(val)
        train=[m+(g,) for m,g in zip(train,trgrid)]
        val=[m+(g,) for m,g in zip(val,vagrid)]
    return train,val,base


def load(arm,device='cpu'):
    dest=OUT/f'fit-{arm}';r=corpus.read(dest/'result.json')
    if r['architecture']!=ARMS[arm] or r['feature_schema']!=(SCHEMA if ARMS[arm]['timing'] else 'base64'):
        raise ValueError('incompatible checkpoint schema')
    if _sha(dest/'flow.pt')!=r['checkpoint_sha256']:raise ValueError('changed checkpoint')
    model=groom.Flow(**r['architecture']).to(device)
    model.load_state_dict(torch.load(dest/'flow.pt',map_location=device,weights_only=False));model.eval()
    return model


def fit(arm):
    kwargs=ARMS[arm];train,val,base=data(kwargs['timing'])
    dest=OUT/f'fit-{arm}';dest.mkdir(exist_ok=True)
    config=dict(arm=arm,architecture=kwargs,feature_schema=SCHEMA if kwargs['timing'] else 'base64',
        seed=20261002,max_updates=36000,prepared_sha256=_sha(OUT/'prepared.json'),
        code={p:_sha(ROOT/p) for p in CODE},validation_selection='first window',
        validation_reporting=['first','middle','end'])
    if (dest/'config.json').exists() and corpus.read(dest/'config.json')!=config:
        raise ValueError('fit identity changed; do not silently restart')
    if (dest/'result.json').exists():load(arm);return corpus.read(dest/'result.json')
    _atomic(dest/'config.json',config)
    torch.manual_seed(20261002);torch.set_num_threads(2);torch.cuda.reset_peak_memory_stats()
    started=time.monotonic()
    history=groom.train_flow(train,val,'cuda',random.Random(20261002),
        out_path=dest/'flow.pt',history_path=dest/'history.json',max_epochs=1200,
        model_kwargs=kwargs,snapshot_updates=(18000,))
    model=groom.Flow(**kwargs).cuda()
    model.load_state_dict(torch.load(dest/'flow.pt',weights_only=False))
    windows={w:dict(zip(('loss','direction_accuracy'),groom.eval_flow(model,val,'cuda',window=w)))
             for w in ('first','middle','end')}
    assert abs(windows['first']['loss']-history['best_val_loss'])<1e-5
    # Same-budget trajectory must reproduce the previous +500 run before the extension.
    if arm=='budget':
        old=corpus.read(OLD/'fit-500/history.json')
        prefix=history['epochs'][:len(old['epochs'])]
        delta=max(abs(a['val_loss']-b['val_loss']) for a,b in zip(prefix,old['epochs']))
        if len(prefix)!=len(old['epochs']) or delta>1e-5:raise ValueError('18k control trajectory changed')
        oldstate=torch.load(OLD/'fit-500/flow.pt',weights_only=False)
        state=torch.load(dest/'flow-18000.pt',weights_only=False)
        if not all(torch.equal(v,state[k]) for k,v in oldstate.items()):raise ValueError('18k control weights changed')
        _atomic(dest/'control-18000.json',dict(max_validation_delta=delta,exact_state_match=True))
    for p,h in base['protected'].items():
        if _sha(ROOT/p)!=h:raise ValueError('production artifact changed')
    result=dict(**config,checkpoint_sha256=_sha(dest/'flow.pt'),validation=windows,
        parameters=sum(p.numel() for p in model.parameters()),updates=history['updates'],
        best_epoch=history['best_epoch'],elapsed_s=time.monotonic()-started,
        peak_cuda_memory_bytes=torch.cuda.max_memory_allocated(),status='TRAINED_QA_PENDING')
    _atomic(dest/'result.json',result)
    return result


def evaluate(arm):
    from qa.comparator_support import _RefValidator
    cache=OUT/f'qa-reference-cache-{arm}.json'
    if not cache.exists():shutil.copyfile(_RefValidator.CACHE_P,cache)
    _RefValidator.CACHE_P=cache
    load(arm)  # Verify the schema and content before using the common evaluator.
    return corpus.evaluate(OUT,arm,'learned',model_kwargs=ARMS[arm])


def reference():
    _,val,_=data(False)
    model=groom.Flow().cuda()
    model.load_state_dict(torch.load(OLD/'fit-500/flow.pt',weights_only=False))
    result=dict(checkpoint_sha256=_sha(OLD/'fit-500/flow.pt'),
        validation={w:dict(zip(('loss','direction_accuracy'),groom.eval_flow(model,val,'cuda',window=w)))
                    for w in ('first','middle','end')})
    _atomic(OUT/'reference-validation.json',result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['prepare','fit','evaluate','reference'])
    p.add_argument('--arm',choices=ARMS,default='budget');a=p.parse_args()
    torch.set_num_threads(2)
    result=(prepare() if a.stage=='prepare' else fit(a.arm) if a.stage=='fit'
            else reference() if a.stage=='reference' else evaluate(a.arm))
    print(json.dumps({k:v for k,v in result.items() if k not in ('ledger','sources','families')},indent=1),flush=True)


if __name__=='__main__':main()
