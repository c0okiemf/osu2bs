"""Combined timing/capacity models on authenticated nested human-map corpora."""
import argparse
import json
import random
import shutil
import time
from pathlib import Path

import torch
import groom
from eval import flow_training_returns as prior
from eval import corpus_scale as corpus
from eval.joint_phrase import ROOT, _sha, _atomic

OUT = ROOT/'experiments/flow-joint-scale-2026-10-02'
ARCH = dict(width=256, timing=True)
BUDGETS = {500:36000, 2000:72000, 2500:83730}
CODE = ('groom.py','timing.py','learned_geometry.py','eval/flow_training_returns.py',
        'eval/flow_joint_scale.py','eval/corpus_scale.py')


def prepare(out=OUT, addition=2000, pool_dir=prior.OLD, clock_cache=None):
    # One full source-clock pass, shared by both nested corpus arms.
    return prior.prepare(out, addition, pool_dir, clock_cache)


def data(addition, out=OUT, pool_dir=prior.OLD):
    prepared=corpus.read(out/'prepared.json')
    train,val,base=prior.source_data(prepared['identity']['addition'],pool_dir)
    if prepared['identity']['pool_sha256']!=_sha(pool_dir/'prepared.json'):
        raise ValueError('frozen corpus changed')
    if prepared['grids_sha256']!=_sha(out/'grids.pt'):
        raise ValueError('clock cache changed')
    trgrid,vagrid=torch.load(out/'grids.pt',weights_only=False)
    assert len(train)==len(trgrid) and len(val)==len(vagrid)
    pool=corpus.read(pool_dir/'prepared.json')['families'][:addition]
    allowed={'new:'+r['id'] for r in pool}
    train=[m+(g,) for m,g in zip(train,trgrid) if not m[6].startswith('new:') or m[6] in allowed]
    val=[m+(g,) for m,g in zip(val,vagrid)]
    assert len({m[6] for m in train})==747+addition
    assert not {m[6] for m in train}&{m[6] for m in val}
    return train,val,base


def load(arm,device='cpu',out=OUT):
    dest=out/f'fit-{arm}';r=corpus.read(dest/'result.json')
    if r['architecture']!=ARCH or r['feature_schema']!=prior.SCHEMA:
        raise ValueError('incompatible checkpoint schema')
    if _sha(dest/'flow.pt')!=r['checkpoint_sha256']:
        raise ValueError('checkpoint changed')
    m=groom.Flow(**ARCH).to(device)
    m.load_state_dict(torch.load(dest/'flow.pt',map_location=device,weights_only=False))
    m.eval();return m


def fit(addition, out=OUT, pool_dir=prior.OLD):
    dest=out/f'fit-{addition}';dest.mkdir(exist_ok=True)
    if (dest/'result.json').exists():
        r=corpus.read(dest/'result.json')
        if r.get('pool_dir',str(prior.OLD))!=str(pool_dir.resolve()) or r['prepared_sha256']!=_sha(out/'prepared.json'):
            raise ValueError('completed fit input identity changed')
        load(str(addition),out=out);return r
    if (dest/'config.json').exists():raise ValueError('partial fit exists; inspect saved state, do not refit silently')
    train,val,base=data(addition,out,pool_dir)
    config=dict(addition=addition,pool_dir=str(pool_dir.resolve()),architecture=ARCH,feature_schema=prior.SCHEMA,seed=20261002,
        max_updates=BUDGETS[addition],early_stop_patience=None,prepared_sha256=_sha(out/'prepared.json'),
        code={p:_sha(ROOT/p) for p in CODE},validation_selection='first window',
        train_families=len({m[6] for m in train}),train_charts=len(train),
        device=torch.cuda.get_device_name(0))
    _atomic(dest/'config.json',config)
    torch.manual_seed(20261002);torch.cuda.reset_peak_memory_stats();started=time.monotonic()
    h=groom.train_flow(train,val,'cuda',random.Random(20261002),out_path=dest/'flow.pt',
        history_path=dest/'history.json',max_epochs=BUDGETS[addition]//30,
        model_kwargs=ARCH,snapshot_updates=(36000,),early_stop_patience=None)
    elapsed=time.monotonic()-started;peak=torch.cuda.max_memory_allocated()
    assert h['updates']==BUDGETS[addition]
    items=[(str(addition),dest/'flow.pt',h)]
    if addition>=2000:
        items.append((str(addition)+'-at36k',dest/'flow-36000.pt',corpus.read(dest/'history-36000.json')))
    for arm,checkpoint,history in items:
        target=out/f'fit-{arm}';target.mkdir(exist_ok=True)
        if checkpoint!=target/'flow.pt':
            shutil.copyfile(checkpoint,target/'flow.pt');_atomic(target/'history.json',history)
        model=groom.Flow(**ARCH).cuda()
        model.load_state_dict(torch.load(target/'flow.pt',weights_only=False))
        windows={w:dict(zip(('loss','direction_accuracy'),groom.eval_flow(model,val,'cuda',window=w)))
                 for w in ('first','middle','end')}
        assert abs(windows['first']['loss']-history['best_val_loss'])<1e-5
        result=dict(config,arm=arm,checkpoint_sha256=_sha(target/'flow.pt'),validation=windows,
            parameters=sum(p.numel() for p in model.parameters()),updates=history['updates'],
            best_epoch=history['best_epoch'],elapsed_s=elapsed if arm==str(addition) else None,
            peak_cuda_memory_bytes=peak if arm==str(addition) else None,status='TRAINED_QA_PENDING')
        _atomic(target/'result.json',result)
        del model
    for p,h in base['protected'].items():
        if _sha(ROOT/p)!=h:raise ValueError('production artifact changed')
    return corpus.read(dest/'result.json')


def evaluate(arm, out=OUT):
    from qa.comparator_support import _RefValidator
    cache=out/f'qa-reference-cache-{arm}.json'
    if not cache.exists():shutil.copyfile(_RefValidator.CACHE_P,cache)
    _RefValidator.CACHE_P=cache
    load(arm,out=out)
    return corpus.evaluate(out,arm,'learned',model_kwargs=ARCH)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['prepare','fit','evaluate'])
    p.add_argument('--arm',choices=['500','2000','2000-at36k','2500','2500-at36k'],default='500')
    p.add_argument('--out',type=Path,default=OUT)
    p.add_argument('--pool',type=Path,default=prior.OLD)
    p.add_argument('--corpus-size',type=int,choices=BUDGETS,default=2000)
    p.add_argument('--clock-cache',type=Path)
    a=p.parse_args();torch.set_num_threads(2)
    if a.stage=='fit' and a.arm.endswith('-at36k'):p.error('36k is a saved checkpoint, not another fit')
    r=(prepare(a.out.resolve(),a.corpus_size,a.pool.resolve(),a.clock_cache.resolve() if a.clock_cache else None) if a.stage=='prepare'
       else fit(int(a.arm),a.out.resolve(),a.pool.resolve()) if a.stage=='fit' else evaluate(a.arm,a.out.resolve()))
    print(json.dumps({k:v for k,v in r.items() if k not in ('ledger','sources','families')},indent=1),flush=True)


if __name__=='__main__':main()
