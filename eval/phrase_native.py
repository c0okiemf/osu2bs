"""Frozen audio-budget planning followed by literal, compatible native phrases."""
from collections import Counter
import json
import time
from unittest.mock import patch

import numpy as np
import torch

from eval import phrase_budget as planner,phrase_state as codec,joint_uniform_audio as control
from eval import joint_compatible as compatible,joint_joins as native,joint_ordered as ordered,joint_pilot as old,joint_approved_fit as approved
from eval.expressive_manifest import freeze_run
from eval.joint_decode import JointState
from eval.joint_phrase import ROOT,_atomic,_sha,encode_events
from eval.joint_export import read_chart
from eval.joint_continuity import describe

OUT=ROOT/'experiments/phrase-native-v1'
SPEC=ROOT/'docs/specs/2026-09-30-budget-native-renderer-design.md'
PROTECTED=('groom.pt','flow.pt','critic.pt','ladder.json','quality_policy.json')


def counts(notes):
    result=[0,0,0]
    for event in encode_events(notes):
        hands=event['hands'];result[2 if all(hands) else 0 if hands[0] else 1]+=1
    return result


def plan(source,audio,model,rate):
    duration=source['duration_beats'];bpm=source['bpm']
    windows=[{'start':float(s),'end':min(float(s)+8,duration)} for s in np.arange(0.,duration,8.)]
    x=np.stack([planner.features(audio,p['start'],p['end'],bpm,rate,duration) for p in windows])
    predicted=planner.predict(model,x,rate,[(p['end']-p['start'])*60/bpm for p in windows],'ridge')
    return [{**p,'counts':c.tolist()} for p,c in zip(windows,predicted)]


def donor_order(bank_counts,requested,audio_distance):
    cost=np.abs(np.asarray(bank_counts)-np.asarray(requested)).sum(1)
    return np.lexsort((np.arange(len(cost)),audio_distance,cost)),cost


def render(source,audio,bank,plans,rate,seed=0):
    expected=[(float(s),min(float(s)+8,source['duration_beats'])) for s in np.arange(0.,source['duration_beats'],8.)]
    if [(p['start'],p['end']) for p in plans]!=expected:raise ValueError('plan does not cover native duration')
    planner.rhythm(plans,source['bpm'])
    state=JointState(source['duration_beats']);used=[];last=[None,None];exits=[None,None]
    rejected=Counter();ledger=[];accepted=[];tested=0
    full_counts=np.array([counts(e['notes']) for e in bank['entries']])
    for p in plans:
        start,end=p['start'],p['end'];width=end-start
        candidates=[e['notes'] if width==8 else [n for n in e['notes'] if n[0]<width] for e in bank['entries']]
        available=full_counts if width==8 else np.array([counts(n) for n in candidates])
        descriptor=old.phrase_descriptor(audio,start,end,source['bpm'],rate*60/source['bpm']);descriptor[-1]=rate*60/source['bpm']
        q=(descriptor-bank['mean'])/bank['sd'];distance=np.square(bank['x']-q).mean(1)
        order,cost=donor_order(available,p['counts'],distance);options=[]
        previous=compatible.last_directional_groups(state.notes())
        for ix in order:
            entry=bank['entries'][ix];key=(entry['family'],entry['start'])
            if used and key==used[-1]:continue
            tested+=1;notes=[(b+start,h,c,l,d) for b,h,c,l,d in candidates[ix]]
            if end<source['duration_beats'] and compatible.terminal_only(entry,notes):
                rejected['nonterminal_unknown_exit']+=1;continue
            if not old._join_ok(previous,notes,source['bpm']):rejected['direction_join']+=1;continue
            checks=native.join_checks(last,exits,notes,entry)
            if any(c['status']!='pass' for c in checks):
                rejected.update(c['status'] for c in checks if c['status']!='pass');continue
            options.append((int(ix),key,notes,entry,checks))
            if len(options)==6:break
        if not options:
            return {'ok':False,'reason':'no_compatible_native_budget_phrase','beat':start,'donors':used,
                    'plans':plans,'budget_ledger':ledger,'join_rejections':dict(rejected),'candidate_checks':tested,'attempt_seed':seed}
        ix,key,notes,entry,checks=options[seed%len(options)]
        ledger.append({**p,'realized_counts':available[ix].tolist(),'absolute_count_error':int(cost[ix]),
                       'minimum_bank_error':int(cost.min()),'minimum_compatible_error':int(cost[options[0][0]]),
                       'prior_hands':codec.canonical(state.last_by_hand),'donor_index':ix})
        for event in encode_events(notes):state.append(event)
        state.rest(end);used.append(key);accepted.extend(checks);native.advance(last,exits,notes,entry)
    return {'ok':bool(state.events),'reason':None if state.events else 'empty_chart',
            'source':{**source,'notes':state.notes(),'events':state.events},'donors':used,'unique_donors':len(set(used)),
            'plans':plans,'budget_ledger':ledger,'join_rejections':dict(rejected),'accepted_join_checks':accepted,
            'candidate_checks':tested,'attempt_seed':seed}


def freeze():
    evaluation=old.evaluation_freeze()
    report=json.loads((planner.OUT/'report.json').read_text());receipt=json.loads((control.OUT/'bank-receipt.json').read_text())
    audit=json.loads((control.OUT/'artifact-audit.json').read_text())
    if report['status']!='PLANNING_SIGNAL_PRESENT' or _sha(planner.OUT/'model.pt')!=report['model_sha256']:raise ValueError('planner not qualified or changed')
    if _sha(control.OUT/'retrieval.pt')!=receipt['bank_sha256'] or _sha(control.OUT/'report.json')!=audit['repeat_report_sha256']:raise ValueError('native control changed')
    views={}
    for family,item in ordered.inputs().items():
        r=receipt['audio_receipts'][family];path=control.OUT/'audio'/family.replace(':','_')/'features.pt'
        if _sha(path)!=r['features_sha256']:raise ValueError('query audio changed')
        views[family]={'proof':item['proof'],'features_sha256':r['features_sha256']}
    return freeze_run(OUT,{'code_sha256':_sha(__file__),'spec_sha256':_sha(SPEC),'planner_report_sha256':_sha(planner.OUT/'report.json'),
        'planner_run_sha256':_sha(planner.OUT/'run.json'),'model_sha256':report['model_sha256'],'bank_receipt':receipt,
        'evaluation_identity':evaluation['identity'],'evaluation_run_sha256':_sha(old.PILOT/'run.json'),
        'control_audit_sha256':_sha(control.OUT/'artifact-audit.json'),'views':views,'protected':{p:_sha(ROOT/p) for p in PROTECTED},
        'dependencies':{p:_sha(ROOT/p) for p in ('eval/phrase_budget.py','eval/joint_compatible.py','eval/joint_joins.py',
            'eval/joint_ordered.py','eval/joint_pilot.py','eval/joint_decode.py','eval/joint_deployment.py','eval/joint_export.py',
            'eval/phrase_state.py','eval/joint_continuity.py','eval/joint_approved_fit.py')},
        'control_artifacts':{str(p.relative_to(ROOT)):_sha(p) for p in (control.OUT/'development').rglob('*')
            if p.is_file() and p.name in ('song.json','record.json','ExpertPlus.dat','Info.dat')}})


def evidence(item,records,chosen,bank,dest):
    audits=[]
    for seed,r in enumerate(records):
        if not r['ok']:audits.append(None);continue
        source=read_chart(dest/'ordered'/str(seed)/'ExpertPlus.dat',dest/'ordered'/str(seed)/'Info.dat')
        source['duration_beats']=item['source']['duration_beats'];a=compatible.audit_ledger(source,r['donors'],bank)
        if a['compressed'] or a['unknown'] or a['nonterminal_unknown_exits']:raise ValueError('native support changed')
        a['gap_diagnostic']=approved.gap_diagnostic(source);audits.append(a)
    key=item['data']['family'].replace(':','_');prior=json.loads((control.OUT/'development'/key/'song.json').read_text())['arms']['ordered']
    def selected(root,arm):
        path=old.PILOT/'development'/key/'b0' if arm['fallback'] else root/'ordered'/str(arm['selected_seed'])
        return describe(read_chart(path/'ExpertPlus.dat',path/'Info.dat'))
    return {'attempts':audits,'continuity_control':selected(control.OUT/'development'/key,prior),'continuity_candidate':selected(dest,chosen)}


def decide(songs,frozen,proofs):
    with patch.object(ordered,'OUT',OUT):report=ordered.decide(songs,frozen)
    all_rows=[r for s in songs for r in s['arms']['ordered']['attempts']]
    complete=[r for r in all_rows if r['ok']];ledger=[p for r in complete for p in r['budget_ledger']]
    report['budget_projection']={'complete_attempts':len(complete),'windows':len(ledger),
        'exact_count_windows':sum(p['absolute_count_error']==0 for p in ledger),
        'mean_absolute_count_error':float(np.mean([p['absolute_count_error'] for p in ledger])) if ledger else None,
        'unreachable_even_before_joins':sum(p['minimum_bank_error']>0 for p in ledger),
        'join_cost_windows':sum(p['minimum_compatible_error']>p['minimum_bank_error'] for p in ledger)}
    report['v15_paired_change']={axis:{s['family']:s['arms']['ordered']['selected']['errors'][axis]['mean']-
        json.loads((control.OUT/'development'/s['family'].replace(':','_')/'song.json').read_text())['arms']['ordered']['selected']['errors'][axis]['mean']
        for s in songs} for axis in ('rhythm','geometry')}
    report['native_evidence']=proofs
    report['budget']='one frozen linear planner; six literal native phrase attempts per deployment family; no model fit or tuning'
    report['arm_semantics']={'ordered':'audio-predicted budgets projected to compatible approved45 native phrases','retrieval':'original deployment R','b0':'protected production control'}
    _atomic(OUT/'report.json',report);return report


def run():
    torch.set_num_threads(2);deadline=time.monotonic()+2700;frozen=freeze()
    bank=torch.load(control.OUT/'retrieval.pt',weights_only=False);model=torch.load(planner.OUT/'model.pt',weights_only=False)
    scales=old.evaluation_freeze()['config']['scales'];kit=old.machine_tools();songs=[];proofs={}
    for family,item in ordered.inputs().items():
        key=family.replace(':','_');dest=OUT/'development'/key;previous=control.OUT/'development'/key
        audio=torch.load(control.OUT/'audio'/key/'features.pt',weights_only=False);plans=plan(item['source'],audio,model,item['rate']);records=[]
        for seed in range(6):
            target=dest/'ordered'/str(seed);record=old.cached_record(target/'record.json',frozen['identity'])
            if record is None:
                if time.monotonic()>=deadline:return {'status':'INCOMPLETE','family':family,'seed':seed}
                started=time.monotonic();attempt=render(item['source'],audio,bank,plans,item['rate'],seed);attempt['elapsed_s']=time.monotonic()-started
                record=ordered.measure(attempt,item['data'],target,frozen['identity'],scales,kit,previous/'ordered'/str(seed)/'record.json')
                print(f'budget native {family} s{seed}: {record["ok"]} {record.get("machine",{}).get("verdict")}',flush=True)
            records.append(record)
        chosen=ordered.select(records,item['b0']);prior=json.loads((previous/'song.json').read_text())
        song={**prior,'identity':frozen['identity'],'arms':{'retrieval':prior['arms']['retrieval'],'ordered':chosen}}
        _atomic(dest/'song.json',song);songs.append(song);proofs[family]=evidence(item,records,chosen,bank,dest)
    freeze();return decide(songs,frozen,proofs)


if __name__=='__main__':print(json.dumps(run(),indent=1))
