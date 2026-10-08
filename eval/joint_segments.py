"""Contiguous32-beat donor units over the quarantined literal eight-beat bank."""
from collections import Counter
import json
import math
import time
from unittest.mock import patch

import numpy as np
import torch

from eval import joint_quarantine as control,joint_compatible as compatible,joint_joins as native,joint_ordered as ordered,joint_pilot as old
from eval.expressive_manifest import freeze_run
from eval.joint_model import _torch_save
from eval.joint_decode import JointState
from eval.joint_phrase import ROOT,_atomic,_sha,encode_events
from eval.joint_export import read_chart
from eval.joint_continuity import describe

OUT=ROOT/'experiments/joint-phrase-v13/contiguous-segments'
SPEC=ROOT/'docs/specs/2026-09-30-contiguous-segments-design.md'
SPAN=32


def segment_bank(pieces,train):
    families={e['family'] for e in pieces['entries']}
    if {d['family'] for d in train}!=families or any(d['fit_role']!='train' for d in train):raise ValueError('segment bank requires exact retained fit-train inventory')
    data={d['family']:d for d in train};lookup={(e['family'],e['start']):e for e in pieces['entries']};entries=[]
    for e in pieces['entries']:
        keys=[(e['family'],e['start']+8*i) for i in range(4)]
        if not all(k in lookup for k in keys):continue
        notes=[(b+8*i,h,c,l,d) for i,k in enumerate(keys) for b,h,c,l,d in lookup[k]['notes']]
        rate=len({(b,h) for b,h,c,l,d in notes})/SPAN;datum=data[e['family']]
        entries.append({'family':e['family'],'start':e['start'],'donors':keys,
            'descriptor':old.phrase_descriptor(datum['audio'],e['start'],e['start']+SPAN,datum['source']['bpm'],rate)})
    if not entries:raise ValueError('no full native segments')
    x=np.stack([e['descriptor'] for e in entries]);mean=x.mean(0);sd=np.maximum(x.std(0),1e-6)
    return {'entries':entries,'mean':mean,'sd':sd,'x':(x-mean)/sd}


def materialize(entry,lookup,start,end,duration,bpm,previous,last,exits):
    notes=[];donors=[];checks=[];reasons=Counter();last=list(last);exits=list(exits)
    for i,key in enumerate(entry['donors']):
        target=start+i*8
        if target>=end:break
        stop=min(target+8,end);piece=lookup[tuple(key)]
        # Add once at final target time: preserve exact eight-beat reconstruction.
        ns=[(b+float(target),h,c,l,d) for b,h,c,l,d in piece['notes'] if b<stop-target]
        if stop<duration and compatible.terminal_only(piece,ns):reasons['nonterminal_unknown_exit']+=1
        cs=native.join_checks(last,exits,ns,piece);checks.extend(cs)
        reasons.update(c['status'] for c in cs if c['status']!='pass')
        native.advance(last,exits,ns,piece);notes.extend(ns);donors.append(key)
    if not old._join_ok(previous,notes,bpm):reasons['direction_join']+=1
    return {'notes':notes,'donors':donors,'checks':checks,'reasons':dict(reasons),'last':last,'exits':exits}


def retrieve(source,audio,bank,pieces,rate,seed=0):
    state=JointState(source['duration_beats']);lookup={(e['family'],e['start']):e for e in pieces['entries']}
    used=[];segments=[];last=[None,None];exits=[None,None];rejected=Counter();accepted=[];ranks=[];tested=0
    for start in np.arange(0.,source['duration_beats'],SPAN):
        end=min(start+SPAN,source['duration_beats']);previous=compatible.last_directional_groups(state.notes())
        q=(old.phrase_descriptor(audio,start,end,source['bpm'],rate*60/source['bpm'])-bank['mean'])/bank['sd']
        order=np.argsort(np.square(bank['x']-q).mean(1),kind='stable');choices=[]
        for rank,ix in enumerate(order,1):
            e=bank['entries'][ix];key=(e['family'],e['start'])
            if (segments and key==segments[-1]) or (used and tuple(e['donors'][0])==tuple(used[-1])):continue
            tested+=1
            m=materialize(e,lookup,float(start),end,source['duration_beats'],source['bpm'],previous,last,exits)
            if m['reasons']:rejected.update(m['reasons']);continue
            choices.append((key,m,rank))
            if len(choices)==6:break
        if not choices:
            return {'ok':False,'reason':'no_compatible_native_segment','beat':float(start),'donors':used,'segments':segments,
                    'join_rejections':dict(rejected),'accepted_join_checks':accepted,'candidate_checks':tested,'selected_raw_ranks':ranks}
        key,m,rank=choices[seed%len(choices)]
        for event in encode_events(m['notes']):state.append(event)
        state.rest(float(end));used.extend(m['donors']);segments.append(key);accepted.extend(m['checks']);ranks.append(rank)
        last,exits=m['last'],m['exits']
    return {'ok':bool(state.events),'reason':None if state.events else 'empty_chart','source':{**source,'notes':state.notes(),'events':state.events},
            'donors':used,'segments':segments,'unique_donors':len(set(map(tuple,used))),'attempt_seed':seed,
            'join_rejections':dict(rejected),'accepted_join_checks':accepted,'candidate_checks':tested,'selected_raw_ranks':ranks}


def audit_ledger(source,record,bank,pieces):
    result=compatible.audit_ledger(source,record['donors'],pieces)
    lookup={(e['family'],e['start']):e for e in bank['entries']};expected=[]
    if len(record['segments'])!=math.ceil(source['duration_beats']/SPAN):raise ValueError('incomplete segment ledger')
    for i,key in enumerate(record['segments']):
        count=math.ceil(min(SPAN,source['duration_beats']-i*SPAN)/8)
        expected.extend(lookup[tuple(key)]['donors'][:count])
    if list(map(tuple,record['donors']))!=list(map(tuple,expected)):raise ValueError('noncontiguous source segment')
    return {**result,'native_segments_exact':True,'segments':len(record['segments']),
            'native_continuations':sum(a[0]==b[0] and a[1]+8==b[1] for a,b in zip(expected,expected[1:])),
            'eight_beat_boundaries':max(0,len(expected)-1)}


def freeze_experiment():
    parent,receipt=control.prepare();report=json.loads((control.OUT/'report.json').read_text())
    if report['identity']!=parent['identity'] or not report['gates']['complete_panel']:raise ValueError('quarantined control incomplete')
    return freeze_run(OUT,{'parent_identity':parent['identity'],'spec_sha256':_sha(SPEC),'code_sha256':_sha(__file__),
        'control_report_sha256':_sha(control.OUT/'report.json'),'control_bank_receipt':receipt,'span_beats':SPAN,
        'excluded_train_families':parent['config']['excluded_train_families'],
        'control_artifacts':{str(p.relative_to(ROOT)):_sha(p) for p in (control.OUT/'development').rglob('*')
            if p.is_file() and p.name in ('song.json','record.json','ExpertPlus.dat','Info.dat')}})


def prepare():
    f=freeze_experiment();path=OUT/'retrieval.pt';rp=OUT/'retrieval-receipt.json'
    if not rp.exists():
        pieces=torch.load(control.OUT/'retrieval.pt',weights_only=False);families={e['family'] for e in pieces['entries']}
        train=[d for d in old.load_data('train') if d['family'] in families]
        bank=segment_bank(pieces,train);_torch_save(path,bank)
        _atomic(rp,{'identity':f['identity'],'sha256':_sha(path),'segments':len(bank['entries']),'families':len({e['family'] for e in bank['entries']})})
    receipt=json.loads(rp.read_text())
    if receipt['identity']!=f['identity'] or receipt['sha256']!=_sha(path):raise ValueError('segment bank changed')
    return f,receipt


def decide(songs,frozen,controls,evidence):
    with patch.object(control,'OUT',OUT):report=control.decide(songs,frozen,controls,evidence)
    report['segment_length_ablation']=report.pop('quarantine_ablation')
    for d in report['segment_length_ablation'].values():
        d['eight_beat_control_mean']=d.pop('overlap_exposed_control_mean');d['thirty_two_beat_mean']=d.pop('quarantined_bank_mean')
    report['arm_semantics']['ordered']='quarantined contiguous32-beat native source segments'
    report['search_cost']['total_native_segment_selections']=report['search_cost'].pop('total_phrase_selections')
    chosen=[e['attempts'][e['selected_seed']] for e in evidence.values() if e['selected_seed'] is not None]
    report['long_context']={'selected_native_continuations':sum(e['native_continuations'] for e in chosen),
        'selected_eight_beat_boundaries':sum(e['eight_beat_boundaries'] for e in chosen),'scope':'nonfallback selected source-continuation retention, not quality'}
    report['budget']='six32-beat-segment attempts per family; quarantined8-beat matched control plus historical B0/R; no fit'
    _atomic(OUT/'report.json',report);return report


def run(deadline_seconds=2700):
    deadline=time.monotonic()+min(deadline_seconds,2700);torch.set_num_threads(4)
    frozen,receipt=prepare();bank=torch.load(OUT/'retrieval.pt',weights_only=False);pieces=torch.load(control.OUT/'retrieval.pt',weights_only=False)
    scales=old.evaluation_freeze()['config']['scales'];kit=old.machine_tools();songs=[];controls={};evidence={}
    for family,item in ordered.inputs().items():
        key=family.replace(':','_');dest=OUT/'development'/key;root=control.OUT/'development'/key
        prior=json.loads((root/'song.json').read_text());controls[family]=prior;records=[];audits=[]
        for seed in range(6):
            target=dest/'ordered'/str(seed);rec=old.cached_record(target/'record.json',frozen['identity'])
            if rec is None:
                if time.monotonic()>=deadline:return {'status':'INCOMPLETE','family':family,'next_seed':seed,'families':len(songs)}
                started=time.monotonic();attempt=retrieve(item['source'],item['data']['audio'],bank,pieces,item['rate'],seed);attempt['elapsed_s']=time.monotonic()-started
                rec=ordered.measure(attempt,item['data'],target,frozen['identity'],scales,kit,root/'ordered'/str(seed)/'record.json')
                print(f'native segments {family} s{seed}: {rec["ok"]} {rec.get("machine",{}).get("verdict")}',flush=True)
            audit=None
            if rec['ok']:
                src=read_chart(target/'ExpertPlus.dat',target/'Info.dat');src['duration_beats']=item['source']['duration_beats']
                audit=audit_ledger(src,rec,bank,pieces)
                if audit['compressed'] or audit['unknown'] or audit['nonterminal_unknown_exits']:raise ValueError('native context contract failed')
            records.append(rec);audits.append(audit)
        chosen=ordered.select(records,item['b0']);song={**prior,'identity':frozen['identity'],'arms':{'retrieval':prior['arms']['retrieval'],'ordered':chosen}}
        _atomic(dest/'song.json',song);songs.append(song)
        before=prior['arms']['ordered'];pp=old.PILOT/'development'/key/'b0' if before['fallback'] else root/'ordered'/str(before['selected_seed'])
        previous=read_chart(pp/'ExpertPlus.dat',pp/'Info.dat');previous['duration_beats']=item['source']['duration_beats']
        cp=old.PILOT/'development'/key/'b0' if chosen['fallback'] else dest/'ordered'/str(chosen['selected_seed'])
        evidence[family]={'attempts':audits,'selected_seed':chosen['selected_seed'],
            'eight_beat_control_audit':None if before['fallback'] else compatible.audit_ledger(previous,before['selected']['donors'],pieces),
            'continuity_control':describe(previous),'continuity_candidate':describe(read_chart(cp/'ExpertPlus.dat',cp/'Info.dat'))}
    prepare();return decide(songs,frozen,controls,evidence)


if __name__=='__main__':print(json.dumps(run(),indent=1))
