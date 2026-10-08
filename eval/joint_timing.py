"""No-fit continuation: condition existing timing proposals on legal support."""
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from eval.joint_decode import JointState, sample_event
from eval.joint_model import (OUT as MODEL, EMPTY, N_GAP, SNAPSHOTS, JointModel,
                              actions, context, decode_gap, literal_slots, source_rate)
from eval.joint_phrase import ROOT, _atomic, _sha
from eval import joint_pilot as old

OUT=ROOT/'experiments/joint-phrase-v2/legal-timing'
SPEC=ROOT/'docs/specs/2026-09-30-legal-timing-continuation-design.md'


def timing_support(cursor, end, kind, residuals):
    """Proposed times and legal categories, before any sampling or modification."""
    residuals=np.asarray(residuals,dtype=float)
    if residuals.shape!=(N_GAP,) or not np.isfinite(residuals).all():
        raise ValueError('invalid_timing_residuals')
    if not math.isfinite(cursor) or not math.isfinite(end) or not 0<=cursor<end \
            or kind not in (0,1,2):
        raise ValueError('invalid_timing_state')
    targets=[end,cursor]+[cursor+decode_gap(c,float(np.clip(residuals[c],-16,8)))
                           for c in range(2,N_GAP)]
    allowed=[True,kind!=1]+[cursor<t<end for t in targets[2:]]
    return targets,allowed


def rollout(model, source, audio, rate, njs=18, seed=0, temperature=1., max_actions=20000):
    model.eval()
    device=next(model.parameters()).device
    if not math.isfinite(temperature) or temperature<=0:
        raise ValueError('invalid temperature')
    gen=torch.Generator().manual_seed(seed)
    state=JointState(source['duration_beats'])
    previous,hidden,kind,last_gap=[EMPTY]*6,None,2,0.
    end=min(8.,state.duration)
    removed=[]
    rest_only=0
    started=time.monotonic()

    def result(ok,reason,step):
        return {'ok':ok,'reason':reason,'source':{**source,'notes':state.notes(),'events':state.events},
                'actions':step,'beat':state.cursor,'attempt_seed':seed,
                'elapsed_s':time.monotonic()-started,
                'timing_support':{'mean_removed_mass':float(np.mean(removed)) if removed else 0.,
                                  'max_removed_mass':max(removed,default=0.),
                                  'rest_only_actions':rest_only,'decisions':len(removed)}}

    with torch.no_grad():
        for step in range(max_actions):
            if state.cursor>=state.duration:
                return result(bool(state.events),None if state.events else 'empty_chart',step)
            ctx=context(audio,state.cursor,source['bpm'],rate,njs,source['walls'],source['bombs'])
            h,hidden=model.hidden(torch.tensor(ctx,device=device)[None,None],
                     torch.tensor(previous,device=device)[None,None],
                     torch.tensor([[kind]],device=device),
                     torch.tensor([[last_gap]],device=device,dtype=torch.float32),hidden)
            h=h[0,0]
            logits=model.gap(h).detach().float().cpu()
            if logits.shape!=(N_GAP,) or not torch.isfinite(logits).all():
                return result(False,'invalid_gap_logits',step)
            try:
                targets,allowed=timing_support(state.cursor,end,kind,
                                                model.residual(h).detach().float().cpu().numpy())
            except ValueError as exc:
                return result(False,str(exc),step)
            allowed=torch.tensor(allowed)
            removed.append(float(torch.softmax(logits/temperature,0)[~allowed].sum()))
            rest_only+=int(allowed.sum()==1)
            legal_logits=logits.masked_fill(~allowed,-float('inf'))
            category=int(torch.multinomial(torch.softmax(legal_logits/temperature,0),1,generator=gen))
            target=targets[category]
            if category==0:
                last_gap=end-state.cursor
                state.rest(end)
                kind=0
                end=min(end+8,state.duration)
                continue
            delta=target-state.cursor
            cat=torch.tensor(category,device=device)
            counts=model.count_logits(h,cat)
            if not torch.isfinite(counts).all():
                return result(False,'invalid_count_logits',step)
            clone=torch.Generator();clone.set_state(gen.get_state())
            sampled_count=int(torch.multinomial(torch.softmax(counts.detach().float().cpu()[1:]/temperature,0),
                                               1,generator=clone))+1

            def slots(hand,slot,prefix):
                tokens=[EMPTY]*6
                for hh,ns in enumerate(prefix):
                    for ss,(c,l,d) in enumerate(ns):
                        tokens[hh*3+ss]=(c*3+l)*9+d
                return model.slot_logits(h,cat,torch.tensor(sampled_count,device=device),
                                         torch.tensor(tokens,device=device))[hand*3+slot]
            try:
                event=sample_event(target,counts,slots,gen,temperature)
                state.append(event)
            except (ValueError,RuntimeError) as exc:
                return result(False,str(exc),step)
            previous=literal_slots(event)
            kind,last_gap=1,delta
    return result(False,'action_budget_exhausted',max_actions)


def freeze_experiment():
    from eval.expressive_manifest import freeze_run
    base=old.evaluation_freeze()
    references={str(p.relative_to(ROOT)):_sha(p) for p in (old.PILOT/'development').glob('**/*')
                if p.is_file() and p.suffix in ('.json','.dat')}
    config={'v1_identity':base['identity'],'spec_sha256':_sha(SPEC),
            'code_sha256':{p:_sha(ROOT/p) for p in ('eval/joint_timing.py','eval/joint_pilot.py',
                'eval/joint_decode.py','eval/joint_export.py','eval/joint_model.py',
                'qa/neighbours.py','convert.py')},
            'checkpoints':{str(step):_sha(MODEL/f'snapshot-{step}.pt') for step in SNAPSHOTS},
            'reference_artifacts':references,'no_fit':True}
    return freeze_run(OUT,config)


def source_audit(deadline_seconds=2700):
    frozen=freeze_experiment()
    path=OUT/'source-audit.json'
    if path.exists():
        report=json.loads(path.read_text())
        if report['identity']!=frozen['identity']:
            raise ValueError('source audit identity changed')
        return report
    deadline=time.monotonic()+min(deadline_seconds,2700)
    reports=[]
    for role in ('train','validation','development'):
        for data in old.load_data(role):
            name=data['family'].replace(':','_')
            saved=OUT/'source-audit'/f'{name}.json'
            if saved.exists():
                report=json.loads(saved.read_text())
                if report['identity']!=frozen['identity']:
                    raise ValueError('source audit identity changed')
                reports.append(report)
                continue
            if time.monotonic()>=deadline:
                return {'status':'INCOMPLETE','families':len(reports)}
            rows=actions(data['source'])
            counts={'events':0,'rests':0,'exact_unsupported':0,'tensor_unsupported':0,'clamped':0}
            witnesses=[];max_error=0.
            for ix,row in enumerate(rows):
                cat=row['gap']
                if cat==0:
                    counts['rests']+=1
                    continue
                counts['events']+=1
                residual=row['residual']
                counts['clamped']+=int(not -16<=residual<=8)
                for precision,res in (('exact',residual),('tensor',float(data['residual'][ix]))):
                    target=row['cursor']+decode_gap(cat,max(-16,min(8,res)))
                    legal=(row['cursor']==target<row['end'] and row['prev_kind']!=1) if cat==1 \
                        else row['cursor']<target<row['end']
                    if not legal:
                        counts[precision+'_unsupported']+=1
                        if len(witnesses)<3:
                            witnesses.append({'action':ix,'precision':precision,'cursor':row['cursor'],
                                              'end':row['end'],'target':target})
                    if precision=='tensor':max_error=max(max_error,abs(target-row['target_beat']))
            report={'identity':frozen['identity'],'family':data['family'],'role':role,**counts,
                    'max_tensor_timing_error_beats':max_error,'witnesses':witnesses}
            _atomic(saved,report);reports.append(report)
    totals={k:sum(r[k] for r in reports) for k in counts}
    result={'identity':frozen['identity'],'status':'SUPPORT_VERIFIED' if not totals['exact_unsupported'] \
            and not totals['tensor_unsupported'] else 'SOURCE_SUPPORT_FAILURE',
            'families':len(reports),'totals':totals,
            'max_tensor_timing_error_beats':max(r['max_tensor_timing_error_beats'] for r in reports)}
    _atomic(path,result)
    return result


def validate(deadline_seconds=2700):
    deadline=time.monotonic()+min(deadline_seconds,2700)
    torch.set_num_threads(2)
    frozen=freeze_experiment()
    audit=json.loads((OUT/'source-audit.json').read_text())
    if audit['identity']!=frozen['identity'] or audit['status']!='SUPPORT_VERIFIED':
        raise ValueError('source timing support not verified')
    base=old.evaluation_freeze()['config']
    data=old.load_data('validation')
    summaries=[]
    for update in SNAPSHOTS:
        model=JointModel().eval()
        model.load_state_dict(torch.load(MODEL/f'snapshot-{update}.pt',map_location='cpu',weights_only=True))
        identity=frozen['identity']+':'+frozen['config']['checkpoints'][str(update)]
        records=[]
        for d in data:
            for seed in (0,1):
                dest=OUT/'validation'/str(update)/d['family'].replace(':','_')/str(seed)
                rec=old.cached_record(dest/'record.json',identity)
                if rec is None:
                    if time.monotonic()>=deadline:
                        return {'status':'INCOMPLETE','snapshot':update,'attempts':len(records)}
                    result=rollout(model,old.generation_source(d),d['audio'],base['validation_rate'],seed=seed)
                    rec=old.measure_attempt(result,d,dest,identity,base['scales'])
                    print(f"validation {update} {d['family']} s{seed}: {rec['ok']} "
                          f"{rec['reason']} actions={rec['actions']} removed={rec['timing_support']['mean_removed_mass']:.4f}",flush=True)
                records.append(rec)
        good=[r for r in records if r['ok']]
        values=[sum(r['errors'][a]['mean'] for a in ('rhythm','geometry')) for r in good
                if all(r['errors'][a]['mean'] is not None for a in ('rhythm','geometry'))]
        from collections import Counter
        summary={'update':update,'snapshot_sha256':frozen['config']['checkpoints'][str(update)],
                 'attempts':len(records),'complete':len(good),'failures':len(records)-len(good),
                 'failure_reasons':dict(Counter(r['reason'] for r in records if not r['ok'])),
                 'error':float(np.mean(values)) if good and len(values)==len(good) else None,
                 'mean_removed_mass':float(np.mean([r['timing_support']['mean_removed_mass'] for r in records]))}
        _atomic(OUT/'validation'/f'summary-{update}.json',summary);summaries.append(summary)
    selected=min(summaries,key=lambda r:(r['failures'],float('inf') if r['error'] is None else r['error'],r['update']))
    result={'identity':frozen['identity'],'status':'VALIDATION_COMPLETE','selected':selected['update'],
            'learned_candidate':selected['update']>0,'snapshots':summaries,
            'selection':'failure count; rhythm+geometry error; earliest update'}
    _atomic(OUT/'selection.json',result)
    return result


def development(deadline_seconds=2700):
    deadline=time.monotonic()+min(deadline_seconds,2700)
    torch.set_num_threads(4)
    frozen=freeze_experiment()
    selection=json.loads((OUT/'selection.json').read_text())
    if selection['identity']!=frozen['identity']:
        raise ValueError('selection identity mismatch')
    model=JointModel().eval()
    model.load_state_dict(torch.load(MODEL/f"snapshot-{selection['selected']}.pt",map_location='cpu',weights_only=True))
    base=old.evaluation_freeze()['config']
    kit=old.machine_tools()
    songs=[]
    for d in old.load_data('development'):
        key=d['family'].replace(':','_')
        original=json.loads((old.PILOT/'development'/key/'song.json').read_text())
        src,transform=old.baseline(d)
        identity=hashlib.sha256(json.dumps({'experiment':frozen['identity'],'baseline':transform,
                         'snapshot':selection['selected']},sort_keys=True).encode()).hexdigest()
        dest=OUT/'development'/key
        records=[]
        for seed,temp in enumerate((.85,1.,1.15,.85,1.,1.15)):
            target=dest/'model'/str(seed)
            rec=old.cached_record(target/'record.json',identity)
            if rec is None:
                if time.monotonic()>=deadline:
                    return {'status':'INCOMPLETE','families':len(songs),'family':d['family'],'next_seed':seed}
                result=rollout(model,old.generation_source(d,src),d['audio'],source_rate(src),
                               seed=seed,temperature=temp)
                rec=old.measure_attempt(result,d,target,identity,base['scales'],kit)
                print(f"development {d['family']} s{seed}: {rec['ok']} {rec['reason']} "
                      f"{rec.get('machine',{}).get('verdict')}",flush=True)
            records.append(rec)
        selected=next((i for i,r in enumerate(records) if r['ok'] and r.get('machine',{}).get('admitted')),None)
        arm={'fallback':selected is None,'selected_seed':selected,
             'selected':original['b0'] if selected is None else records[selected],
             'attempts':[{'ok':r['ok'],'reason':r['reason'],'verdict':r.get('machine',{}).get('verdict'),
                          'emitted_notes':r.get('emitted_notes',0)} for r in records]}
        song={**original,'identity':identity,'arms':{'model':arm,'retrieval':original['arms']['retrieval']}}
        _atomic(dest/'song.json',song);songs.append(song)
    freeze_experiment()
    return old.decision(songs,selection,frozen,output_root=OUT)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['audit','validate','development'])
    parser.add_argument('--deadline-seconds',type=float,default=2700)
    args=parser.parse_args()
    result={'audit':source_audit,'validate':validate,'development':development}[args.command](args.deadline_seconds)
    print(json.dumps(result,indent=1))
