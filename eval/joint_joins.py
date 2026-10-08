"""Literal phrase retrieval with source-relative temporal join support."""
import bisect
from collections import Counter
import json
import math
import time
from unittest.mock import patch

import numpy as np
import torch

from eval import joint_workload as workload,joint_ordered as ordered,joint_pilot as old
from eval.joint_model import _torch_save
from eval.joint_decode import JointState
from eval.joint_phrase import ROOT,_atomic,_sha,encode_events
from eval.joint_export import read_chart
from eval.joint_continuity import describe

OUT=ROOT/'experiments/joint-phrase-v9/native-context-joins'
SPEC=ROOT/'docs/specs/2026-09-30-native-context-joins-design.md'


def enrich_bank(base,sources):
    if set(sources)!={e['family'] for e in base['entries']}:raise ValueError('native source inventory mismatch')
    times={fam:[sorted({b for b,h,c,l,d in src['notes'] if h==hand}) for hand in (0,1)] for fam,src in sources.items()}
    entries=[]
    for e in base['entries']:
        before=[];after=[]
        for ts in times[e['family']]:
            a=bisect.bisect_left(ts,e['start']);b=bisect.bisect_left(ts,e['start']+8)
            before.append(ts[a]-ts[a-1] if a<b and a>0 else None)
            after.append(ts[b]-ts[b-1] if a<b and b<len(ts) else None)
        entries.append({**e,'entry_gap_beats':before,'exit_gap_beats':after})
    return {**base,'entries':entries}


def join_checks(last,exit_gaps,notes,entry):
    checks=[]
    for hand in (0,1):
        times=[b for b,h,c,l,d in notes if h==hand]
        if not times or last[hand] is None:continue
        actual=min(times)-last[hand];before=exit_gaps[hand];after=entry['entry_gap_beats'][hand]
        known=all(v is not None and math.isfinite(v) and v>0 for v in (before,after))
        minimum=min(before,after) if known else None
        status='native_context_unknown' if not known else 'native_gap_compression' if actual+1e-9<minimum else 'pass'
        checks.append({'hand':hand,'previous_beat':last[hand],'next_beat':min(times),'actual_gap_beats':actual,
                       'native_exit_gap_beats':before,'native_entry_gap_beats':after,'minimum_gap_beats':minimum,'status':status})
    return checks


def advance(last,exit_gaps,notes,entry):
    for hand in (0,1):
        times=[b for b,h,c,l,d in notes if h==hand]
        if times:last[hand]=max(times);exit_gaps[hand]=entry['exit_gap_beats'][hand]


def retrieve(source,audio,bank,rate,seed=0):
    state=JointState(source['duration_beats']);used=[];last=[None,None];exits=[None,None];rejected=Counter();accepted=[]
    for start in np.arange(0.,source['duration_beats'],8.):
        end=min(start+8,source['duration_beats'])
        q=(old.phrase_descriptor(audio,start,end,source['bpm'],rate*60/source['bpm'])-bank['mean'])/bank['sd']
        order=np.argsort(np.square(bank['x']-q).mean(1),kind='stable')[:32]
        compatible=[];previous=state.notes()
        for ix in order:
            entry=bank['entries'][ix];key=(entry['family'],entry['start'])
            if used and key==used[-1]:continue
            notes=[(b+float(start),h,c,l,d) for b,h,c,l,d in entry['notes'] if b<end-start]
            if not old._join_ok(previous,notes,source['bpm']):rejected['direction_join']+=1;continue
            checks=join_checks(last,exits,notes,entry)
            if any(c['status']!='pass' for c in checks):
                rejected.update(c['status'] for c in checks if c['status']!='pass');continue
            compatible.append((key,notes,entry,checks))
            if len(compatible)==6:break
        if not compatible:
            return {'ok':False,'reason':'no_native_supported_join','beat':float(start),'donors':used,
                    'join_rejections':dict(rejected),'accepted_join_checks':accepted}
        key,notes,entry,checks=compatible[seed%len(compatible)]
        for event in encode_events(notes):state.append(event)
        state.rest(float(end));used.append(key);accepted.extend(checks);advance(last,exits,notes,entry)
    return {'ok':bool(state.events),'reason':None if state.events else 'empty_chart',
            'source':{**source,'notes':state.notes(),'events':state.events},'donors':used,'unique_donors':len(set(used)),
            'attempt_seed':seed,'join_rejections':dict(rejected),'accepted_join_checks':accepted}


def audit_ledger(source,donors,bank):
    if len(donors)!=math.ceil(source['duration_beats']/8):raise ValueError('incomplete donor ledger')
    lookup={(e['family'],e['start']):e for e in bank['entries']};expected=[];last=[None,None];exits=[None,None];checks=[]
    for i,key in enumerate(donors):
        entry=lookup[tuple(key)];start=i*8;end=min(start+8,source['duration_beats'])
        notes=[(b+float(start),h,c,l,d) for b,h,c,l,d in entry['notes'] if b<end-start]
        checks.extend(join_checks(last,exits,notes,entry));advance(last,exits,notes,entry);expected.extend(notes)
    if sorted(map(tuple,source['notes']))!=sorted(expected):raise ValueError('literal donor reconstruction changed')
    bad=[c for c in checks if c['status']!='pass']
    return {'literal_reconstruction_exact':True,'checked_joins':len(checks),
            'compressed':sum(c['status']=='native_gap_compression' for c in checks),
            'unknown':sum(c['status']=='native_context_unknown' for c in checks),
            'minimum_join_gap_ms':min((c['actual_gap_beats']*60000/source['bpm'] for c in checks),default=None),
            'failing_examples':bad[:12]}


def native_sources():
    run=json.loads((old.OUT/'run.json').read_text());sources={};hashes={}
    for r in run['config']['records']:
        if r['fit_role']!='train':continue
        path=ROOT/'experiments/joint-phrase-v1/readiness/sources'/(r['family'].replace(':','_')+'.json')
        if _sha(path)!=r['artifact_sha256']:raise ValueError('native source artifact changed')
        sources[r['family']]=json.loads(path.read_text());hashes[str(path.relative_to(ROOT))]=r['artifact_sha256']
    if len(sources)!=316:raise ValueError('native fit-train inventory changed')
    return sources,hashes


def freeze_experiment():
    from eval.expressive_manifest import freeze_run
    parent,bank=workload.prepare();report=json.loads((workload.OUT/'report.json').read_text())
    if report['identity']!=parent['identity'] or not report['gates']['complete_panel']:raise ValueError('workload comparison incomplete')
    _,hashes=native_sources()
    return freeze_run(OUT,{'parent_identity':parent['identity'],'spec_sha256':_sha(SPEC),'code_sha256':_sha(__file__),
             'control_report_sha256':_sha(workload.OUT/'report.json'),'control_bank_receipt':bank,'native_sources':hashes,
             'control_artifacts':{str(p.relative_to(ROOT)):_sha(p) for p in (workload.OUT/'development').rglob('*')
                  if p.is_file() and p.name in ('song.json','record.json','ExpertPlus.dat','Info.dat')}})


def prepare():
    f=freeze_experiment();path=OUT/'retrieval.pt';receipt_path=OUT/'retrieval-receipt.json'
    if not receipt_path.exists():
        base=torch.load(workload.OUT/'retrieval.pt',weights_only=False);sources,_=native_sources()
        bank=enrich_bank(base,sources);_torch_save(path,bank)
        _atomic(receipt_path,{'identity':f['identity'],'sha256':_sha(path),'entries':len(bank['entries'])})
    receipt=json.loads(receipt_path.read_text())
    if receipt['identity']!=f['identity'] or receipt['sha256']!=_sha(path):raise ValueError('native-edge bank changed')
    return f,receipt


def decide(songs,frozen,controls,evidence):
    with patch.object(ordered,'OUT',OUT):report=ordered.decide(songs,frozen)
    ablation={}
    for axis in ('rhythm','geometry'):
        before=[controls[s['family']]['arms']['ordered']['selected']['errors'][axis]['mean'] for s in songs]
        after=[s['arms']['ordered']['selected']['errors'][axis]['mean'] for s in songs]
        ablation[axis]={'workload_control_mean':float(np.mean(before)),'native_join_mean':float(np.mean(after)),
                       'paired_differences':{s['family']:b-a for s,a,b in zip(songs,before,after)}}
    complete=[a for e in evidence.values() for a in e['attempts'] if a is not None]
    report.update(arm_semantics={'ordered':'workload-corrected original retrieval plus native temporal join support',
                                 'retrieval':'frozen original deployment retrieval','b0':'original production baseline'},
                  native_join_ablation=ablation,join_evidence=evidence,
                  construction_contract={'complete_candidates':len(complete),'all_complete_pass':bool(complete) and all(a['compressed']==a['unknown']==0 for a in complete),
                       'scope':'nonfallback retrieved candidates only; passing this contract does not imply a quality pass'},
                  budget='six new candidates per family; cached v8 workload, v7 original retrieval and B0 controls; no fits')
    _atomic(OUT/'report.json',report);return report


def run(deadline_seconds=2700):
    deadline=time.monotonic()+min(deadline_seconds,2700);torch.set_num_threads(4)
    frozen,receipt=prepare();bank=torch.load(OUT/'retrieval.pt',weights_only=False)
    scales=old.evaluation_freeze()['config']['scales'];kit=old.machine_tools();songs=[];controls={};evidence={}
    for family,item in ordered.inputs().items():
        key=family.replace(':','_');dest=OUT/'development'/key;root=workload.OUT/'development'/key
        control=json.loads((root/'song.json').read_text());controls[family]=control;records=[];audits=[]
        for seed in range(6):
            target=dest/'ordered'/str(seed);rec=old.cached_record(target/'record.json',frozen['identity'])
            if rec is None:
                if time.monotonic()>=deadline:return {'status':'INCOMPLETE','family':family,'next_seed':seed,'families':len(songs)}
                started=time.monotonic();attempt=retrieve(item['source'],item['data']['audio'],bank,item['rate'],seed)
                attempt['elapsed_s']=time.monotonic()-started
                rec=ordered.measure(attempt,item['data'],target,frozen['identity'],scales,kit,root/'ordered'/str(seed)/'record.json')
                print(f'native joins {family} s{seed}: {rec["ok"]} {rec.get("machine",{}).get("verdict")}',flush=True)
            audit=None
            if rec['ok']:
                src=read_chart(target/'ExpertPlus.dat',target/'Info.dat');src['duration_beats']=item['source']['duration_beats']
                audit=audit_ledger(src,rec['donors'],bank)
                if audit['compressed'] or audit['unknown']:raise ValueError('exported join contract failed')
            records.append(rec);audits.append(audit)
        chosen=ordered.select(records,item['b0']);song={**control,'identity':frozen['identity'],
                    'arms':{'retrieval':control['arms']['retrieval'],'ordered':chosen}}
        _atomic(dest/'song.json',song);songs.append(song)
        before=control['arms']['ordered'];prevpath=old.PILOT/'development'/key/'b0' if before['fallback'] else root/'ordered'/str(before['selected_seed'])
        prev=read_chart(prevpath/'ExpertPlus.dat',prevpath/'Info.dat');prev['duration_beats']=item['source']['duration_beats']
        prev_audit=None if before['fallback'] else audit_ledger(prev,before['selected']['donors'],bank)
        newpath=old.PILOT/'development'/key/'b0' if chosen['fallback'] else dest/'ordered'/str(chosen['selected_seed'])
        evidence[family]={'attempts':audits,'selected_seed':chosen['selected_seed'],'workload_control_join_audit':prev_audit,
                          'continuity_control':describe(prev),'continuity_candidate':describe(read_chart(newpath/'ExpertPlus.dat',newpath/'Info.dat'))}
    prepare();return decide(songs,frozen,controls,evidence)


if __name__=='__main__':print(json.dumps(run(),indent=1))
