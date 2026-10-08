"""One controlled fit restoring direct supervision at actual song starts."""
import hashlib
import json
from pathlib import Path
import random
import time
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F

from eval.joint_model import (EMPTY,KEYS,SEED,SNAPSHOTS,JointModel,_torch_save,source_rate)
from eval.joint_phrase import ROOT,_atomic,_sha,encode_events
from eval import joint_pilot as old
from eval import joint_timing as timing

OUT=ROOT/'experiments/joint-phrase-v3/start-supervision'
SPEC=ROOT/'docs/specs/2026-09-30-song-start-supervision-design.md'


def crop(row,start):
    if not 0<=start<=len(row['gap'])-128:
        raise ValueError('invalid crop start')
    result={k:row[k][start:start+128] for k in KEYS}
    result['valid']=torch.ones(128,dtype=torch.bool)
    if start>0:result['valid'][:32]=False
    return result


def losses(out,batch):
    valid=batch['valid']
    if not valid.any():raise ValueError('no supervised targets')
    event=valid&(batch['gap']>0)
    positive=valid&(batch['gap']>1)
    value=F.cross_entropy(out['gap'][valid],batch['gap'][valid])
    if positive.any():
        pred=out['residual'].gather(-1,batch['gap'][...,None]).squeeze(-1)
        value=value+F.smooth_l1_loss(pred[positive],batch['residual'][positive])
    if event.any():value=value+F.cross_entropy(out['count'][event],batch['count'][event])
    slots=event[...,None]&(batch['slots']!=EMPTY)
    if slots.any():value=value+F.cross_entropy(out['slots'][slots],batch['slots'][slots])
    return value


def freeze_experiment():
    from eval.expressive_manifest import freeze_run
    parent=timing.freeze_experiment()
    config={'parent_identity':parent['identity'],'spec_sha256':_sha(SPEC),
            'code_sha256':_sha(__file__),
            'comparison_sha256':{p:_sha(timing.OUT/p) for p in ('selection.json','report.json')},
            'data_receipts':{p.name:_sha(p) for p in (old.OUT/'data').glob('*.json')},
            'seed':SEED,'training':{'updates':6000,'snapshots':SNAPSHOTS,'batch':16,
                'explicit_bos_rows':4,'uniform_rows':12,'sequence':128,
                'non_bos_warmup':32,'lr':3e-4,'weight_decay':.01}}
    if len(config['data_receipts'])!=356:raise ValueError('incomplete prepared data')
    return freeze_run(OUT,config)


def train(deadline_seconds=2700):
    deadline=time.monotonic()+min(deadline_seconds,2700)
    frozen=freeze_experiment()
    torch.set_num_threads(4)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data=old.load_data('train')
    rows=[{k:d[k] for k in KEYS} for d in data]
    if len(rows)!=316 or min(len(r['gap']) for r in rows)<128:
        raise ValueError('fit inventory/crop mismatch')
    del data
    random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED)
    model=JointModel().to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.01)
    cursor,bos_crops,history=0,0,[]
    path=OUT/'train_state.pt'
    if path.exists():
        state=torch.load(path,map_location=device,weights_only=False)
        if state['identity']!=frozen['identity']:raise ValueError('training identity changed')
        model.load_state_dict(state['model']);opt.load_state_dict(state['optimizer'])
        random.setstate(state['python_rng']);np.random.set_state(state['numpy_rng'])
        torch.set_rng_state(state['torch_rng'].cpu())
        if torch.cuda.is_available():torch.cuda.set_rng_state_all([x.cpu() for x in state['cuda_rng']])
        cursor,bos_crops,history=state['update'],state['bos_crops'],state['history']
    if cursor==0 and not (OUT/'snapshot-0.pt').exists():
        _torch_save(OUT/'snapshot-0.pt',{k:v.cpu() for k,v in model.state_dict().items()})
    model.train()
    while cursor<6000 and time.monotonic()<deadline:
        crops=[]
        for i in range(16):
            row=random.choice(rows)
            start=0 if i<4 else random.randrange(len(row['gap'])-128+1)
            bos_crops+=int(start==0)
            crops.append(crop(row,start))
        batch={k:torch.stack([c[k] for c in crops]).to(device) for k in (*KEYS,'valid')}
        opt.zero_grad()
        out,_=model(batch)
        loss=losses(out,batch)
        if not torch.isfinite(loss):raise ValueError(f'nonfinite loss at {cursor}')
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step()
        cursor+=1
        if cursor%100==0:
            history.append({'update':cursor,'loss':float(loss),'bos_crops':bos_crops})
            print(f'update {cursor}/6000 loss={float(loss):.4f} BOS_crops={bos_crops}',flush=True)
        if cursor in SNAPSHOTS:
            _torch_save(OUT/f'snapshot-{cursor}.pt',{k:v.cpu() for k,v in model.state_dict().items()})
        if cursor%100==0 or cursor==6000 or time.monotonic()>=deadline:
            _torch_save(path,{'identity':frozen['identity'],'update':cursor,'bos_crops':bos_crops,
                'model':model.state_dict(),'optimizer':opt.state_dict(),'python_rng':random.getstate(),
                'numpy_rng':np.random.get_state(),'torch_rng':torch.get_rng_state(),
                'cuda_rng':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],'history':history})
            _atomic(OUT/'history.json',{'update':cursor,'bos_crops':bos_crops,
                     'bos_supervised_targets':bos_crops*128,'history':history})
    return {'status':'FIT_COMPLETE' if cursor==6000 else 'INCOMPLETE','update':cursor,'bos_crops':bos_crops}


def attempts(model,data,source,rate,dest,identity,temps,deadline,scales,kit=None):
    records=[]
    for seed,temp in enumerate(temps):
        target=dest/str(seed)
        record=old.cached_record(target/'record.json',identity)
        if record is None:
            if time.monotonic()>=deadline:return None
            result=timing.rollout(model,source,data['audio'],rate,seed=seed,temperature=temp)
            record=old.measure_attempt(result,data,target,identity,scales,kit)
            print(f"{dest.parent.name}/{dest.name} {data['family']} s{seed}: {record['ok']} "
                  f"{record['reason']} {record.get('machine',{}).get('verdict')}",flush=True)
        records.append(record)
    return records


def validate(deadline_seconds=2700):
    deadline=time.monotonic()+min(deadline_seconds,2700)
    torch.set_num_threads(2)
    frozen=freeze_experiment();base=old.evaluation_freeze()['config']
    data=old.load_data('validation');summaries=[]
    for update in SNAPSHOTS:
        path=OUT/f'snapshot-{update}.pt'
        if not path.exists():return {'status':'WAITING_FOR_SNAPSHOT','update':update}
        model=JointModel().eval();model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True))
        identity=frozen['identity']+':'+_sha(path)
        records=[]
        for d in data:
            result=attempts(model,d,old.generation_source(d),base['validation_rate'],
                    OUT/'validation'/str(update)/d['family'].replace(':','_'),identity,
                    (1.,1.),deadline,base['scales'])
            if result is None:return {'status':'INCOMPLETE','update':update,'attempts':len(records)}
            records+=result
        good=[r for r in records if r['ok']]
        values=[sum(r['errors'][a]['mean'] for a in ('rhythm','geometry')) for r in good
                if all(r['errors'][a]['mean'] is not None for a in ('rhythm','geometry'))]
        summary={'update':update,'snapshot_sha256':_sha(path),'attempts':len(records),
                 'complete':len(good),'failures':len(records)-len(good),
                 'failure_reasons':dict(Counter(r['reason'] for r in records if not r['ok'])),
                 'error':float(np.mean(values)) if good and len(values)==len(good) else None}
        _atomic(OUT/'validation'/f'summary-{update}.json',summary);summaries.append(summary)
    selected=min(summaries,key=lambda r:(r['failures'],float('inf') if r['error'] is None else r['error'],r['update']))
    result={'status':'VALIDATION_COMPLETE','identity':frozen['identity'],'snapshots':summaries,
            'selected':selected['update'],'learned_candidate':selected['update']>0,
            'selection':'failure count; rhythm+geometry error; earliest update'}
    _atomic(OUT/'selection.json',result)
    return result


def development(deadline_seconds=2700):
    deadline=time.monotonic()+min(deadline_seconds,2700)
    torch.set_num_threads(4)
    frozen=freeze_experiment();base=old.evaluation_freeze()['config']
    selection=json.loads((OUT/'selection.json').read_text())
    if selection['identity']!=frozen['identity']:raise ValueError('selection changed')
    path=OUT/f"snapshot-{selection['selected']}.pt"
    model=JointModel().eval();model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True))
    kit=old.machine_tools();songs=[]
    for d in old.load_data('development'):
        key=d['family'].replace(':','_');src,transform=old.baseline(d)
        original=json.loads((old.PILOT/'development'/key/'song.json').read_text())
        identity=hashlib.sha256(json.dumps({'experiment':frozen['identity'],'baseline':transform,
                            'snapshot_sha256':_sha(path)},sort_keys=True).encode()).hexdigest()
        dest=OUT/'development'/key
        records=attempts(model,d,old.generation_source(d,src),source_rate(src),dest/'model',identity,
                          (.85,1.,1.15,.85,1.,1.15),deadline,base['scales'],kit)
        if records is None:return {'status':'INCOMPLETE','families':len(songs),'family':d['family']}
        selected=next((i for i,r in enumerate(records) if r['ok'] and r.get('machine',{}).get('admitted')),None)
        arm={'selected_seed':selected,'fallback':selected is None,
             'selected':original['b0'] if selected is None else records[selected],
             'attempts':[{'ok':r['ok'],'reason':r['reason'],'verdict':r.get('machine',{}).get('verdict'),
                          'emitted_notes':r.get('emitted_notes',0)} for r in records]}
        song={**original,'identity':identity,'arms':{'model':arm,'retrieval':original['arms']['retrieval']}}
        _atomic(dest/'song.json',song);songs.append(song)
    freeze_experiment()
    return old.decision(songs,selection,frozen,output_root=OUT)


def startup_support(root):
    """Diagnostic only; same 32-action prefix definition for both experiments."""
    from eval.joint_export import read_chart
    from eval.joint_model import actions
    data={d['family']:d for d in old.load_data('development')}
    rows=[]
    for path in sorted((Path(root)/'development').glob('*/model/*/record.json')):
        r=json.loads(path.read_text());machine=r.get('machine')
        if not machine:continue
        src=read_chart(path.parent/'ExpertPlus.dat',path.parent/'Info.dat')
        src['duration_beats']=data[r['family']]['source']['duration_beats']
        src['events']=encode_events(src['notes']);act=actions(src)
        end=act[min(31,len(act)-1)]['target_beat']*60/src['bpm']
        heads=machine['support']['per_head'];row={'family':r['family'],'seed':r['attempt_seed'],
                                               'prefix_end_seconds':end,'verdict':machine['verdict']}
        for name,predicate in [('prefix',lambda h:h['t']<=end),('later',lambda h:h['t']>end)]:
            part=[h for h in heads if predicate(h)]
            row[name+'_heads']=len(part);row[name+'_unsupported']=sum(not h['supported'] for h in part)
        rows.append(row)
    result={'completed_candidates':len(rows),'rows':rows}
    for region in ('prefix','later'):
        total=sum(r[region+'_heads'] for r in rows);bad=sum(r[region+'_unsupported'] for r in rows)
        result[region]={'heads':total,'unsupported':bad,'share_unsupported':bad/total if total else None}
    return result


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['train','validate','development','diagnose'])
    parser.add_argument('--deadline-seconds',type=float,default=2700)
    args=parser.parse_args()
    if args.command=='diagnose':
        result={'v2':startup_support(timing.OUT),'v3':startup_support(OUT)}
        _atomic(OUT/'startup-support.json',result)
    else:result={'train':train,'validate':validate,'development':development}[args.command](args.deadline_seconds)
    print(json.dumps(result,indent=1))
