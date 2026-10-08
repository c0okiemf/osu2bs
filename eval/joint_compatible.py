"""Search the frozen compatible donor domain without weakening native joins."""
from collections import Counter
import json
import time
from unittest.mock import patch

import numpy as np
import torch

from eval import joint_joins as native,joint_ordered as ordered,joint_pilot as old
from eval.joint_decode import JointState
from eval.joint_phrase import ROOT,_atomic,_sha,encode_events
from eval.joint_export import read_chart
from eval.joint_continuity import describe

OUT=ROOT/'experiments/joint-phrase-v10/compatible-search'
SPEC=ROOT/'docs/specs/2026-09-30-compatible-retrieval-search-design.md'


def last_directional_groups(notes):
    result=[]
    for hand in (0,1):
        heads=[n for n in notes if n[1]==hand and n[4]!=8]
        if heads:result.extend(n for n in heads if n[0]==heads[-1][0])
    return result


def terminal_only(entry,notes):
    return any(any(n[1]==hand for n in notes) and
               (entry['exit_gap_beats'][hand] is None or not np.isfinite(entry['exit_gap_beats'][hand]) or entry['exit_gap_beats'][hand]<=0)
               for hand in (0,1))


def retrieve(source,audio,bank,rate,seed=0):
    state=JointState(source['duration_beats']);used=[];last=[None,None];exits=[None,None]
    rejected=Counter();accepted=[];ranks=[];tested=0
    for start in np.arange(0.,source['duration_beats'],8.):
        end=min(start+8,source['duration_beats'])
        q=(old.phrase_descriptor(audio,start,end,source['bpm'],rate*60/source['bpm'])-bank['mean'])/bank['sd']
        order=np.argsort(np.square(bank['x']-q).mean(1),kind='stable')
        compatible=[];previous=last_directional_groups(state.notes())
        for rank,ix in enumerate(order,1):
            entry=bank['entries'][ix];key=(entry['family'],entry['start'])
            if used and key==used[-1]:continue
            tested+=1
            notes=[(b+float(start),h,c,l,d) for b,h,c,l,d in entry['notes'] if b<end-start]
            if end<source['duration_beats'] and terminal_only(entry,notes):
                rejected['nonterminal_unknown_exit']+=1;continue
            if not old._join_ok(previous,notes,source['bpm']):rejected['direction_join']+=1;continue
            checks=native.join_checks(last,exits,notes,entry)
            if any(c['status']!='pass' for c in checks):
                rejected.update(c['status'] for c in checks if c['status']!='pass');continue
            compatible.append((key,notes,entry,checks,rank))
            if len(compatible)==6:break
        if not compatible:
            return {'ok':False,'reason':'no_compatible_donor_in_bank','beat':float(start),'donors':used,
                    'join_rejections':dict(rejected),'accepted_join_checks':accepted,'candidate_checks':tested,
                    'selected_raw_ranks':ranks}
        key,notes,entry,checks,rank=compatible[seed%len(compatible)]
        for event in encode_events(notes):state.append(event)
        state.rest(float(end));used.append(key);accepted.extend(checks);ranks.append(rank)
        native.advance(last,exits,notes,entry)
    return {'ok':bool(state.events),'reason':None if state.events else 'empty_chart',
            'source':{**source,'notes':state.notes(),'events':state.events},'donors':used,'unique_donors':len(set(used)),
            'attempt_seed':seed,'join_rejections':dict(rejected),'accepted_join_checks':accepted,
            'candidate_checks':tested,'selected_raw_ranks':ranks}


def audit_ledger(source,donors,bank):
    result=native.audit_ledger(source,donors,bank)
    lookup={(e['family'],e['start']):e for e in bank['entries']}
    result['nonterminal_unknown_exits']=sum(terminal_only(lookup[tuple(key)],lookup[tuple(key)]['notes']) for key in donors[:-1])
    return result


def freeze_experiment():
    from eval.expressive_manifest import freeze_run
    parent,bank=native.prepare();report=json.loads((native.OUT/'report.json').read_text())
    if report['identity']!=parent['identity'] or not report['gates']['complete_panel']:raise ValueError('native control incomplete')
    return freeze_run(OUT,{'parent_identity':parent['identity'],'spec_sha256':_sha(SPEC),'code_sha256':_sha(__file__),
             'control_report_sha256':_sha(native.OUT/'report.json'),'bank_receipt':bank,
             'failure_audit_sha256':_sha(native.OUT/'failure-search-audit.json'),
             'control_artifacts':{str(p.relative_to(ROOT)):_sha(p) for p in (native.OUT/'development').rglob('*')
                  if p.is_file() and p.name in ('song.json','record.json','ExpertPlus.dat','Info.dat')}})


def decide(songs,frozen,controls,evidence):
    with patch.object(native,'OUT',OUT):report=native.decide(songs,frozen,controls,evidence)
    report['compatibility_search_ablation']=report.pop('native_join_ablation')
    for d in report['compatibility_search_ablation'].values():
        d['native_shortlist_control_mean']=d.pop('workload_control_mean')
        d['compatible_search_mean']=d.pop('native_join_mean')
    complete=[a for e in evidence.values() for a in e['attempts'] if a is not None]
    report['construction_contract']['all_nonterminal_contexts_known']=bool(complete) and all(a['nonterminal_unknown_exits']==0 for a in complete)
    report['arm_semantics']['ordered']='compatible-domain original retrieval with corrected workload and native temporal joins'
    attempts=[r for s in songs for r in s['arms']['ordered']['attempts']]
    ranks=[x for r in attempts for x in r['selected_raw_ranks']]
    report['search_cost']={'candidate_checks':sum(r['candidate_checks'] for r in attempts),
        'selected_rank_median':float(np.median(ranks)) if ranks else None,'selected_rank_max':max(ranks,default=None),
        'selections_beyond_original32':sum(x>32 for x in ranks),'total_phrase_selections':len(ranks),
        'generation_seconds':sum(r['elapsed_s'] for r in attempts)}
    report['budget']='six new chart attempts per family; larger internal bank search disclosed; cached v9/v8/v7/B0 controls, no fit'
    _atomic(OUT/'report.json',report);return report


def run(deadline_seconds=2700):
    deadline=time.monotonic()+min(deadline_seconds,2700);torch.set_num_threads(4)
    frozen=freeze_experiment();bank=torch.load(native.OUT/'retrieval.pt',weights_only=False)
    scales=old.evaluation_freeze()['config']['scales'];kit=old.machine_tools();songs=[];controls={};evidence={}
    for family,item in ordered.inputs().items():
        key=family.replace(':','_');dest=OUT/'development'/key;root=native.OUT/'development'/key
        control=json.loads((root/'song.json').read_text());controls[family]=control;records=[];audits=[]
        for seed in range(6):
            target=dest/'ordered'/str(seed);rec=old.cached_record(target/'record.json',frozen['identity'])
            if rec is None:
                if time.monotonic()>=deadline:return {'status':'INCOMPLETE','family':family,'next_seed':seed,'families':len(songs)}
                started=time.monotonic();attempt=retrieve(item['source'],item['data']['audio'],bank,item['rate'],seed)
                attempt['elapsed_s']=time.monotonic()-started
                rec=ordered.measure(attempt,item['data'],target,frozen['identity'],scales,kit,root/'ordered'/str(seed)/'record.json')
                print(f'compatible search {family} s{seed}: {rec["ok"]} {rec.get("machine",{}).get("verdict")}',flush=True)
            audit=None
            if rec['ok']:
                src=read_chart(target/'ExpertPlus.dat',target/'Info.dat');src['duration_beats']=item['source']['duration_beats']
                audit=audit_ledger(src,rec['donors'],bank)
                if audit['compressed'] or audit['unknown'] or audit['nonterminal_unknown_exits']:raise ValueError('exported native context contract failed')
            records.append(rec);audits.append(audit)
        chosen=ordered.select(records,item['b0']);song={**control,'identity':frozen['identity'],
                    'arms':{'retrieval':control['arms']['retrieval'],'ordered':chosen}}
        _atomic(dest/'song.json',song);songs.append(song)
        before=control['arms']['ordered'];prevpath=old.PILOT/'development'/key/'b0' if before['fallback'] else root/'ordered'/str(before['selected_seed'])
        prev=read_chart(prevpath/'ExpertPlus.dat',prevpath/'Info.dat');prev['duration_beats']=item['source']['duration_beats']
        newpath=old.PILOT/'development'/key/'b0' if chosen['fallback'] else dest/'ordered'/str(chosen['selected_seed'])
        evidence[family]={'attempts':audits,'selected_seed':chosen['selected_seed'],
             'native_shortlist_control_audit':None if before['fallback'] else audit_ledger(prev,before['selected']['donors'],bank),
             'continuity_control':describe(prev),'continuity_candidate':describe(read_chart(newpath/'ExpertPlus.dat',newpath/'Info.dat'))}
    freeze_experiment();return decide(songs,frozen,controls,evidence)


if __name__=='__main__':print(json.dumps(run(),indent=1))
