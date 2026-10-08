"""Keep positive timing probability beyond a boundary as no-note REST probability."""
from collections import Counter
import json
import math
import time
from unittest.mock import patch

import numpy as np
import torch

from eval import joint_approved_fit as fit,joint_timing as timing,joint_timing_tail_audit as tail,joint_pilot as old,joint_ordered as ordered
from eval.joint_decode import JointState,sample_event
from eval.joint_model import EMPTY,N_GAP,JointModel,context,decode_gap,literal_slots
from eval.joint_phrase import ROOT,_atomic,_sha
from eval.joint_export import read_chart
from eval.joint_continuity import describe
from eval.expressive_manifest import freeze_run

OUT=ROOT/'experiments/joint-phrase-v17/censored-timing'
SPEC=ROOT/'docs/specs/2026-09-30-censored-timing-design.md'


def censored_logits(logits,targets,allowed,end,temperature):
    logits=torch.as_tensor(logits).detach().float().cpu()
    if logits.shape!=(N_GAP,) or not torch.isfinite(logits).all() or not math.isfinite(temperature) or temperature<=0:
        raise ValueError('invalid timing probabilities')
    allowed=torch.tensor(allowed,dtype=torch.bool)
    if len(targets)!=N_GAP or allowed.shape!=(N_GAP,) or not allowed[0] or not all(math.isfinite(t) for t in targets):
        raise ValueError('invalid timing support')
    overflow=torch.tensor([i>=2 and t>=end for i,t in enumerate(targets)])
    if (overflow&allowed).any():raise ValueError('overflow timing marked legal')
    scaled=logits/temperature;result=scaled.masked_fill(~allowed,-float('inf'))
    if overflow.any():result[0]=torch.logsumexp(torch.cat([scaled[:1],scaled[overflow]]),0)
    return result,overflow


def rollout(model, source, audio, rate, njs=18, seed=0, temperature=1., max_actions=20000):
    model.eval()
    device=next(model.parameters()).device
    if not math.isfinite(temperature) or temperature<=0:
        raise ValueError('invalid temperature')
    gen=torch.Generator().manual_seed(seed)
    state=JointState(source['duration_beats'])
    previous,hidden,kind,last_gap=[EMPTY]*6,None,2,0.
    end=min(8.,state.duration)
    removed=[];overflow_masses=[];ledger=[]
    rest_only=0
    started=time.monotonic()

    def result(ok,reason,step):
        return {'ok':ok,'reason':reason,'source':{**source,'notes':state.notes(),'events':state.events},
                'actions':step,'beat':state.cursor,'attempt_seed':seed,
                'elapsed_s':time.monotonic()-started,
                'timing_decisions':ledger,
                'boundary_censoring':{'mean_overflow_mass':float(np.mean(overflow_masses)) if overflow_masses else 0.,'max_overflow_mass':max(overflow_masses,default=0.)},
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
                residuals=model.residual(h).detach().float().cpu().numpy()
                targets,allowed=timing.timing_support(state.cursor,end,kind,residuals)
            except ValueError as exc:
                return result(False,str(exc),step)
            legal_logits,overflow=censored_logits(logits,targets,allowed,end,temperature)
            raw=torch.softmax(logits/temperature,0)
            removed.append(float(raw[~torch.tensor(allowed)&~overflow].sum()))
            overflow_masses.append(float(raw[overflow].sum()))
            rest_only+=int(torch.isfinite(legal_logits).sum()==1)
            category=int(torch.multinomial(torch.softmax(legal_logits,0),1,generator=gen))
            ledger.append({'cursor':state.cursor,'end':end,'kind':kind,'category':category,
                           'target':targets[category],'residual':float(residuals[category])})
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
    parent=fit.freeze_experiment();report=json.loads((fit.OUT/'report.json').read_text());selection=json.loads((fit.OUT/'selection.json').read_text())
    if report['identity']!=parent['identity'] or not report['gates']['complete_panel']:raise ValueError('matched fit control incomplete')
    snapshot=fit.OUT/f"snapshot-{selection['selected']}.pt"
    if selection['identity']!=parent['identity'] or _sha(snapshot)!=selection['snapshot_sha256']:raise ValueError('selected checkpoint changed')
    frozen=freeze_run(OUT,{'parent_identity':parent['identity'],'code_sha256':_sha(__file__),'spec_sha256':_sha(SPEC),
        'selection_sha256':_sha(fit.OUT/'selection.json'),'snapshot_sha256':_sha(snapshot),'prepared_sha256':_sha(fit.OUT/'prepared.json'),
        'tail_audit_sha256':_sha(tail.OUT/'report.json'),'tail_code_sha256':_sha(tail.__file__),'control_report_sha256':_sha(fit.OUT/'report.json'),
        'control_artifacts':{str(p.relative_to(ROOT)):_sha(p) for root in (fit.OUT/'development',fit.OUT/'validation'/str(selection['selected']))
            for p in root.rglob('*') if p.is_file() and p.name in ('record.json','song.json','ExpertPlus.dat','Info.dat')}})
    return frozen,parent,selection


def load_model(selection):
    model=JointModel().eval();model.load_state_dict(torch.load(fit.OUT/f"snapshot-{selection['selected']}.pt",map_location='cpu',weights_only=True));return model


def raw_probe(frozen,parent,selection,prepared,model):
    target=OUT/'raw-boundary-probe.json'
    if target.exists():
        result=json.loads(target.read_text())
        if result['identity']!=frozen['identity']:raise ValueError('probe identity changed')
        return result
    rows=[];exports=0
    for family in sorted(fit.VALIDATION):
        data=torch.load(ROOT/parent['config']['sources'][family]['payload'],weights_only=False);audio=fit.load_view(family,parent)
        for seed in range(2):
            raw=[];hook=model.gap.register_forward_hook(lambda m,args,out:raw.append(out.detach().softmax(-1).numpy().copy()))
            try:result,trace=tail.replay(model,old.generation_source(data),audio,prepared['validation_rate'],seed)
            finally:hook.remove()
            root=fit.OUT/'validation'/str(selection['selected'])/family.replace(':','_')/str(seed)
            actual=read_chart(root/'ExpertPlus.dat',root/'Info.dat');assert result['source']['notes']==actual['notes']
            assert len(trace)==len(raw);exports+=1
            for row,p in zip(trace,raw):
                if row['tiny_same_hand']:rows.append({**row,'family':family,'seed':seed,'raw_probability':float(p[row['category']]),'raw_mode':int(p.argmax()),'raw_rest':float(p[0])})
    result={'identity':frozen['identity'],'exact_replayed_exports':exports,'tiny_decisions':len(rows),
        'raw_probability_quantiles':np.quantile([r['raw_probability'] for r in rows],[0,.1,.5,.9,1]).tolist(),
        'raw_modal':sum(r['raw_mode']==r['category'] for r in rows),'legal_modal':sum(r['rank']==1 for r in rows),'rows':rows}
    _atomic(target,result);return result


def audit_targets(record,source):
    events=iter(source['events'] if 'events' in source else fit.encode_events(source['notes']));cursor=0.;count=0
    for row in record['timing_decisions']:
        assert row['cursor']==cursor
        cat=row['category'];target=row['target']
        if cat==0:assert target==row['end']
        else:
            expected=cursor if cat==1 else cursor+decode_gap(cat,float(np.clip(row['residual'],-16,8)))
            assert target==expected and target<row['end']
            event=next(events);assert event['beat']==target;count+=1
        cursor=target
    assert next(events,None) is None
    if record['ok']:assert cursor==source['duration_beats']
    return {'events':count,'decisions':len(record['timing_decisions']),'raw_event_targets_exact':True}


def validate(deadline_seconds=2700):
    deadline=time.monotonic()+min(deadline_seconds,2700);torch.set_num_threads(2);frozen,parent,selection=freeze_experiment()
    prepared=json.loads((fit.OUT/'prepared.json').read_text());model=load_model(selection)
    probe=raw_probe(frozen,parent,selection,prepared,model);rows=[]
    for family in sorted(fit.VALIDATION):
        data=torch.load(ROOT/parent['config']['sources'][family]['payload'],weights_only=False);audio=fit.load_view(family,parent)
        for seed in range(2):
            dest=OUT/'validation'/family.replace(':','_')/str(seed);rec=old.cached_record(dest/'record.json',frozen['identity'])
            if rec is None:
                if time.monotonic()>=deadline:return {'status':'INCOMPLETE','family':family,'seed':seed}
                attempt=rollout(model,old.generation_source(data),audio,prepared['validation_rate'],seed=seed)
                rec=old.measure_attempt(attempt,data,dest,frozen['identity'],prepared['validation_scales'])
                print('censored validation',family,seed,rec['ok'],rec['reason'],flush=True)
            priorroot=fit.OUT/'validation'/str(selection['selected'])/family.replace(':','_')/str(seed)
            prior=json.loads((priorroot/'record.json').read_text());gaps=None;audit=None
            if rec['ok']:
                actual=read_chart(dest/'ExpertPlus.dat',dest/'Info.dat');actual['duration_beats']=data['source']['duration_beats']
                audit=audit_targets(rec,actual);gaps=fit.gap_diagnostic(actual)
            rows.append({'family':family,'seed':seed,'ok':rec['ok'],'reason':rec['reason'],'errors':rec.get('errors'),
                'control_errors':prior.get('errors'),'gaps':gaps,'control_gaps':fit.gap_diagnostic(read_chart(priorroot/'ExpertPlus.dat',priorroot/'Info.dat')),'target_audit':audit})
    result={'identity':frozen['identity'],'status':'VALIDATION_DIAGNOSTIC_COMPLETE','rows':rows,'raw_probe_sha256':_sha(OUT/'raw-boundary-probe.json'),
        'selected_checkpoint':selection['selected'],'selection':'none; fixed parent checkpoint and decoder','complete':sum(r['ok'] for r in rows)}
    _atomic(OUT/'validation-report.json',result);return result


def decide(songs,frozen,selection,evidence):
    with patch.object(fit,'OUT',OUT):report=fit.decide(songs,frozen,selection,evidence)
    differences={}
    for axis in ('rhythm','geometry'):
        paired={}
        for song in songs:
            control=json.loads((fit.OUT/'development'/song['family'].replace(':','_')/'song.json').read_text())
            a=song['arms']['ordered']['selected']['errors'][axis]['mean'];b=control['arms']['ordered']['selected']['errors'][axis]['mean']
            paired[song['family']]=a-b if a is not None and b is not None else None
        differences[axis]={'paired_differences':paired,'mean_difference':float(np.mean(list(paired.values()))) if all(x is not None for x in paired.values()) else None}
    report['boundary_decoder_ablation']=differences;report['arm_semantics']['ordered']='same approved45 checkpoint; overflow probability assigned to no-note REST'
    report.update(budget='no fit;12 diagnostic validation and48 development attempts; fixed parent checkpoint',
                  raw_probe_sha256=_sha(OUT/'raw-boundary-probe.json'),validation_report_sha256=_sha(OUT/'validation-report.json'))
    _atomic(OUT/'report.json',report);return report


def development(deadline_seconds=2700):
    deadline=time.monotonic()+min(deadline_seconds,2700);torch.set_num_threads(2);frozen,parent,selection=freeze_experiment()
    validation=json.loads((OUT/'validation-report.json').read_text())
    if validation['identity']!=frozen['identity'] or len(validation['rows'])!=12:raise ValueError('validation diagnostic incomplete')
    model=load_model(selection);kit=old.machine_tools();scales=old.evaluation_freeze()['config']['scales'];songs=[];evidence={}
    for family,item in ordered.inputs().items():
        key=family.replace(':','_');root=OUT/'development'/key;priorroot=fit.OUT/'development'/key
        prior=json.loads((priorroot/'song.json').read_text());audio=fit.load_view(family,parent);records=[];audits=[]
        for seed,temp in enumerate((.85,1.,1.15,.85,1.,1.15)):
            target=root/'ordered'/str(seed);rec=old.cached_record(target/'record.json',frozen['identity'])
            if rec is None:
                if time.monotonic()>=deadline:return {'status':'INCOMPLETE','family':family,'seed':seed}
                attempt=rollout(model,item['source'],audio,item['rate'],seed=seed,temperature=temp)
                rec=ordered.measure(attempt,item['data'],target,frozen['identity'],scales,kit,priorroot/'ordered'/str(seed)/'record.json')
                print('censored development',family,seed,rec['ok'],rec.get('machine',{}).get('verdict',rec['reason']),flush=True)
            audit=None
            if rec['ok']:
                actual=read_chart(target/'ExpertPlus.dat',target/'Info.dat');actual['duration_beats']=item['source']['duration_beats'];audit=audit_targets(rec,actual)
            records.append(rec);audits.append(audit)
        chosen=ordered.select(records,item['b0']);song={**prior,'identity':frozen['identity'],'arms':{'retrieval':prior['arms']['retrieval'],'ordered':chosen}}
        _atomic(root/'song.json',song);songs.append(song)
        cp=old.PILOT/'development'/key/'b0' if chosen['fallback'] else root/'ordered'/str(chosen['selected_seed'])
        actual=read_chart(cp/'ExpertPlus.dat',cp/'Info.dat')
        evidence[family]={'selected':describe(actual),'selected_gaps':fit.gap_diagnostic(actual),'target_audits':audits,
            'attempt_gaps':[fit.gap_diagnostic(read_chart(root/'ordered'/str(i)/'ExpertPlus.dat',root/'ordered'/str(i)/'Info.dat')) if r['ok'] else None for i,r in enumerate(records)]}
    freeze_experiment();return decide(songs,frozen,selection,evidence)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=['validate','development']);args=parser.parse_args()
    print(json.dumps({'validate':validate,'development':development}[args.command](),indent=1))
