"""Bounded exact-onset / learned-hand experiment on the authenticated old corpus."""
import argparse
from collections import Counter, defaultdict
import gc
import json
from pathlib import Path
import random
import time

import torch
from torch.nn import functional as F
import groom
import onset_flow as model_code
from eval.joint_phrase import ROOT, _sha, _atomic
from eval import corpus_scale
from eval.flow_joint_scale import data

OUT = ROOT/'experiments/onset-flow-2026-10-08'
SOURCE = ROOT/'experiments/flow-joint-scale-2026-10-02'
CODE = ('onset_flow.py','learned_geometry.py','groom.py','parity.py','swing_clearance.py','eval/onset_flow.py')


def prepare(out=OUT):
    out.mkdir(parents=True, exist_ok=True)
    identity = dict(schema=model_code.SCHEMA, source_prepared_sha256=_sha(SOURCE/'prepared.json'),
                    code={p:_sha(ROOT/p) for p in CODE})
    receipt=out/'prepared.json'
    if receipt.exists():
        r=corpus_scale.read(receipt)
        if r['identity'] != identity or r['dataset_sha256'] != _sha(out/'dataset.pt'):
            raise ValueError('prepared onset dataset changed')
        return r
    old=corpus_scale.read(SOURCE/'prepared.json')
    for p,h in old['sources'].items():
        if _sha(Path(p)) != h:raise ValueError('human-map source changed: '+p)
    train,val,base=data(2000, out=SOURCE)
    assert not {m[6] for m in train}&{m[6] for m in val}
    outputs=[];stats=[]
    for name,rows in [('train',train),('val',val)]:
        mass=defaultdict(float);surv=defaultdict(float)
        for m in rows:
            mass[m[6]]+=m[5]
            if len(m[2]) > groom.CTX or (name=='val' and len(m[2])==groom.CTX):surv[m[6]]+=m[5]
        packed=[]
        for i,m in enumerate(rows):
            if len(m[2]) > groom.CTX or (name=='val' and len(m[2])==groom.CTX):
                for events in (m[2],groom.mirror_events(m[2])):
                    x,y=model_code.training_sequence(events,m[7])
                    packed.append((x,y,m[5]*mass[m[6]]/surv[m[6]]/2,m[6]))
            rows[i]=None
            if (i+1)%500==0:print('prepared',name,i+1,flush=True)
        stats.append(dict(split=name,charts_with_mirrors=len(packed),families=len(surv),
                          onsets=sum(len(x) for x,*_ in packed),
                          short_sequences_padded=sum(len(x)<groom.CTX for x,*_ in packed),
                          family_sampling_mass=sum(mass[f] for f in surv)))
        outputs.append(packed)
    torch.save(outputs,out/'dataset.pt')
    result=dict(identity=identity,dataset_sha256=_sha(out/'dataset.pt'),splits=stats,
                source_hashes_checked=len(old['sources']),protected=base['protected'])
    _atomic(receipt,result);print(json.dumps(result,indent=1),flush=True);return result


def crop(x,y,start=0):
    x,y=x[start:start+groom.CTX],y[start:start+groom.CTX]
    n=groom.CTX-len(x)
    return F.pad(x,(0,0,0,n)),F.pad(y,(0,0,0,n),value=-100)


def validation_batch(val,window,device):
    rows=[]
    for x,y,*_ in val:
        end=max(0,len(x)-groom.CTX)
        rows.append(crop(x,y,{'first':0,'middle':end//2,'end':end}[window]))
    return tuple(torch.stack([r[j] for r in rows]).to(device) for j in (0,1))


@torch.inference_mode()
def measure(model,x,y,details=False):
    model.eval();value=float(model_code.loss(model,x,y))
    modes,_=model(x,y);pred=modes.argmax(-1);mask=y[...,0]>=0
    truth=y[...,0][mask];pred=pred[mask]
    r=dict(loss=value,mode_accuracy=float((truth==pred).float().mean()))
    if details:
        r['mode_confusion']=torch.bincount(truth*3+pred,minlength=9).reshape(3,3).cpu().tolist()
        r['mode_counts']=torch.bincount(truth,minlength=3).cpu().tolist()
        r['majority_mode_accuracy']=max(r['mode_counts'])/len(truth)
    return r


def train(out=OUT,updates=72000):
    dest=out/'fit';dest.mkdir(exist_ok=True)
    if (dest/'config.json').exists():raise ValueError('fit already exists; never overwrite or silently retry')
    prepared=corpus_scale.read(out/'prepared.json')
    if prepared['dataset_sha256'] != _sha(out/'dataset.pt'):raise ValueError('dataset changed')
    for p,h in prepared['identity']['code'].items():
        if _sha(ROOT/p)!=h:raise ValueError('training code changed since preparation: '+p)
    for p,h in prepared['protected'].items():
        if _sha(ROOT/p)!=h:raise ValueError('production artifact changed')
    tr,val=torch.load(out/'dataset.pt',weights_only=False)
    device='cuda';torch.manual_seed(20261008);rng=random.Random(20261008)
    model=model_code.OnsetFlow().to(device);optimizer=torch.optim.Adam(model.parameters(),lr=3e-4)
    config=dict(schema=model_code.SCHEMA,updates=updates,seed=20261008,width=256,
        parameters=sum(p.numel() for p in model.parameters()),device=torch.cuda.get_device_name(0),
        prepared_sha256=_sha(out/'prepared.json'),code={p:_sha(ROOT/p) for p in CODE},
        selection='first validation window, no early stop',batch=8,context=groom.CTX)
    _atomic(dest/'config.json',config)
    vx,vy=validation_batch(val,'first',device)
    weights=[r[2] for r in tr];best=float('inf');best_state=None;best_step=None
    history=[];started=time.monotonic();torch.cuda.reset_peak_memory_stats()
    for step in range(1,updates+1):
        model.train();batch=[]
        for x,y,*_ in rng.choices(tr,weights=weights,k=8):
            batch.append(crop(x,y,rng.randrange(max(1,len(x)-groom.CTX+1))))
        bx,by=(torch.stack([r[j] for r in batch]).to(device) for j in (0,1))
        value=model_code.loss(model,bx,by)
        if not torch.isfinite(value):raise ValueError('non-finite training loss')
        optimizer.zero_grad();value.backward();optimizer.step()
        if step%30==0 or step==updates:
            result=measure(model,vx,vy)
            if not __import__('math').isfinite(result['loss']):raise ValueError('non-finite validation loss')
            history.append(dict(step=step,**result))
            if result['loss'] < best-1e-4:
                best=result['loss'];best_step=step
                best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            if step%300==0:print('step',step,'loss',round(result['loss'],5),'hands',round(result['mode_accuracy'],4),flush=True)
        if step==36000 or step==updates:
            arm=out/f'fit-{step}';arm.mkdir(exist_ok=True)
            torch.save(dict(schema=model_code.SCHEMA,width=256,state=best_state),arm/'model.pt')
            _atomic(arm/'result.json',dict(config,updates=step,best_step=best_step,best_loss=best,
                checkpoint_sha256=_sha(arm/'model.pt'),status='TRAINED_QA_PENDING'))
            _atomic(arm/'history.json',history)
    elapsed=time.monotonic()-started;peak=torch.cuda.max_memory_allocated()
    del tr,model,optimizer,vx,vy;gc.collect();torch.cuda.empty_cache()
    for step in sorted({36000,updates}):
        arm=out/f'fit-{step}'
        if not arm.exists():continue
        model=model_code.load_model(arm/'model.pt',device)
        windows={}
        for window in ('first','middle','end'):
            x,y=validation_batch(val,window,device);windows[window]=measure(model,x,y,True)
            del x,y
        result=corpus_scale.read(arm/'result.json')
        result.update(validation=windows,elapsed_s=elapsed if step==updates else None,
                      peak_cuda_memory_bytes=peak if step==updates else None)
        _atomic(arm/'result.json',result);del model
    _atomic(dest/'complete.json',dict(updates=updates,elapsed_s=elapsed,peak_cuda_memory_bytes=peak))


def evaluate(out=OUT,step=72000):
    import convert,motion
    from eval.joint_export import export_chart
    from eval.joint_pilot import machine_tools,machine_measure
    from eval.joint_continuity import describe
    from eval.quality_metrics import four_gram_stats
    from swing_clearance import blocked_approaches
    from qa.comparator_support import _RefValidator
    import shutil
    original_cache=_RefValidator.CACHE_P
    _RefValidator.CACHE_P=out/f'qa-cache-{step}.json'
    if not _RefValidator.CACHE_P.exists() and original_cache.exists():
        shutil.copyfile(original_cache,_RefValidator.CACHE_P)
    dest=out/f'evaluation/{step}';dest.mkdir(parents=True,exist_ok=True)
    checkpoint=out/f'fit-{step}/model.pt';model=model_code.load_model(checkpoint)
    config=dict(checkpoint_sha256=_sha(checkpoint),code={p:_sha(ROOT/p) for p in (*CODE,'convert.py')},seeds=[0,1])
    if (dest/'config.json').exists() and corpus_scale.read(dest/'config.json')!=config:
        raise ValueError('evaluation identity changed')
    _atomic(dest/'config.json',config)
    kit=None;rows=[]
    for entry in corpus_scale.read(ROOT/'eval/clean_rhythm_panel.json')['entries']:
        for kind in ('osu','audio'):
            assert _sha(ROOT/entry[kind])==entry[kind+'_sha256']
        meta,objects,bpm,_=convert.parse_osu(ROOT/entry['osu'])
        onsets=model_code.source_onsets(objects,meta['_timing'],bpm)
        for seed in (0,1):
            directory=dest/entry['song']/str(seed);rp=directory/'record.json'
            if rp.exists():
                row=corpus_scale.read(rp)
                for p,h in row.get('artifacts',{}).items():assert _sha(directory/p)==h
                rows.append(row);continue
            row=dict(song=entry['song'],seed=seed,source_onsets=len(onsets),source_osu_sha256=entry['osu_sha256'])
            began=time.monotonic()
            try:
                notes=model_code.decode(onsets,model,seed)
                row['decode_elapsed_s']=time.monotonic()-began
                assert {n['t'] for n in notes}=={r['t'] for r in onsets}
                problems=convert.check(notes,bpm,cap=float('inf'))
                if problems:raise ValueError('; '.join(problems))
                ms=[(n['t'],n['hand'],n['col'],n['layer'],n['dir']) for n in notes]
                source=dict(title=entry['song'],bpm=bpm,notes=[(t*bpm/60000,h,c,l,d) for t,h,c,l,d in ms],
                            bombs=[],walls=[],duration_beats=onsets[-1]['t']*bpm/60000+1)
                emitted=export_chart(source,directory)
                assert not blocked_approaches(emitted['notes'])
                # Also exercise the real rounded v2 exporter, not only evaluation v3.
                convert.export_charts({'ExpertPlus':(notes,[],convert.diff_spec('ExpertPlus'))},bpm,meta,directory/'rounded')
                rounded=corpus_scale.read(directory/'rounded/ExpertPlus.dat')['_notes']
                rbpm=corpus_scale.read(directory/'rounded/Info.dat')['_beatsPerMinute']
                max_error=max(abs(n['t']-r['_time']*60000/rbpm) for n,r in zip(notes,rounded))
                assert max_error<=.5e-5*60000/rbpm+1e-7
                if kit is None:print('loading machine QA',flush=True);kit=machine_tools()
                heads=[n for n in notes if n['dir']!=8]
                hands=Counter(n['hand'] for n in heads)
                row.update(ok=True,notes=len(notes),heads=len(heads),hand_counts=dict(hands),
                    doubles=sum(count==2 for count in Counter(n['t'] for n in heads).values()),
                    onset_times_preserved=True,max_export_error_ms=max_error,
                    peak_nps=convert.peak_nps([n['t'] for n in heads]),
                    density_policy_problems=convert.check(notes,bpm),
                    machine=machine_measure(emitted,kit),motion=motion.report(ms),
                    repetition=four_gram_stats(ms,60000/bpm),continuity=describe(source),
                    artifacts={p:_sha(directory/p) for p in ('ExpertPlus.dat','Info.dat','rounded/ExpertPlus.dat','rounded/Info.dat')})
            except ValueError as exc:row.update(ok=False,reason=str(exc))
            row['elapsed_s']=time.monotonic()-began
            _atomic(directory/'source-onsets.json',onsets);_atomic(rp,row);rows.append(row)
            print('evaluated',step,entry['song'],seed,row['ok'],row.get('reason',''),flush=True)
    _atomic(dest/'report.json',dict(config=config,attempts=len(rows),valid=sum(r['ok'] for r in rows),
        qa=dict(Counter(r.get('machine',{}).get('verdict','NOT_EVALUATED') for r in rows)),families=rows,
        status='COMPLETE_USER_REVIEW_PENDING'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['prepare','train','evaluate'])
    p.add_argument('--out',type=Path,default=OUT)
    p.add_argument('--step',type=int,choices=[36000,72000],default=72000)
    a=p.parse_args();torch.set_num_threads(2)
    if a.stage=='prepare':prepare(a.out.resolve())
    elif a.stage=='train':train(a.out.resolve())
    else:evaluate(a.out.resolve(),a.step)


if __name__=='__main__':main()
