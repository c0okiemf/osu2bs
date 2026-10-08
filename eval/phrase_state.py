"""Lossless phrase-budget labels and explicit causal histories; no generation or fit."""
from collections import Counter
import hashlib
import json

import torch

from eval import joint_diverse_fit as parent,joint_rhythm_failure_audit as failure
from eval.expressive_manifest import freeze_run
from eval.joint_model import EMPTY,actions,action_notes
from eval.joint_phrase import ROOT,_atomic,_sha,encode_events,phrase_windows,verify_sources

OUT=ROOT/'experiments/phrase-state-reachability-v1'
SPEC=ROOT/'docs/specs/2026-09-30-phrase-state-reachability-design.md'


def canonical(value):
    return json.loads(json.dumps(value,sort_keys=True,allow_nan=False))


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def annotate(plans,rows):
    if not plans:raise ValueError('missing phrase plans')
    for plan in plans:
        if len(plan['counts'])!=3 or any(type(n) is not int or n<0 for n in plan['counts']):raise ValueError('invalid phrase budget')
    window=0;remaining=list(plans[0]['counts']);last=[None,None];result=[]
    for row in rows:
        if window>=len(plans) or row['end']!=plans[window]['end']:raise ValueError('action/phrase boundary mismatch')
        cursor=row['cursor']
        state={'window':window,'remaining':remaining.copy(),'hands':canonical(last),
               'age_beats':[None if hand is None else cursor-hand['beat'] for hand in last]}
        if any(age is not None and age<0 for age in state['age_beats']):raise ValueError('history from the future')
        result.append({'action':canonical(row),'state':state})
        if row['gap']==0:
            if any(remaining) or row['target_beat']!=plans[window]['end']:raise ValueError('REST before budget consumption')
            window+=1
            if window<len(plans):remaining=list(plans[window]['counts'])
            continue
        notes=action_notes([row]);event=encode_events(notes)[0];hands=event['hands']
        kind=2 if all(hands) else 0 if hands[0] else 1
        remaining[kind]-=1
        if remaining[kind]<0:raise ValueError('phrase budget exhausted')
        for hand,group in enumerate(hands):
            if group:last[hand]={'beat':event['beat'],'notes':[list(n) for n in group]}
    if window!=len(plans):raise ValueError('unconsumed final phrase')
    return result


def encode(source):
    events=encode_events(source['notes']);source={**source,'events':events}
    windows=phrase_windows(source['notes'],source['bombs'],source['walls'],source['duration_beats']);plans=[]
    for window in windows:
        counts=[0,0,0]
        for event in window['events']:
            hands=event['hands'];counts[2 if all(hands) else 0 if hands[0] else 1]+=1
        plans.append({'start':window['start'],'end':window['end'],'counts':counts})
    metadata=canonical({k:source[k] for k in ('bpm','duration_beats','bombs','walls','authored')})
    scene={**metadata,'notes':canonical(source['notes'])}
    return {'schema':1,'scene_sha256':digest(scene),'metadata':metadata,'plans':plans,'steps':annotate(plans,actions(source))}


def decode(packet):
    if packet.get('schema')!=1:raise ValueError('unsupported phrase-state schema')
    notes=action_notes([step['action'] for step in packet['steps']])
    source={**packet['metadata'],'notes':notes,'events':encode_events(notes)}
    if digest({**packet['metadata'],'notes':canonical(notes)})!=packet['scene_sha256']:raise ValueError('literal source scene changed')
    expected=encode(source)
    if packet!=expected:raise ValueError('phrase budget, action or causal state changed')
    return source


def coverage(packet):
    absent=0;actions_absent=0;carried=0;stacks=Counter();events=0
    for step in packet['steps']:
        row,state=step['action'],step['state'];missing=0
        for hand,prior in enumerate(state['hands']):
            if prior is not None and all(t==EMPTY for t in row['prev_slots'][hand*3:hand*3+3]):missing+=1
            if prior is not None and prior['beat']<packet['plans'][state['window']]['start']:carried+=1
        absent+=missing;actions_absent+=int(missing>0)
        if row['gap']:
            events+=1
            for hand in (0,1):
                size=sum(t!=EMPTY for t in row['slots'][hand*3:hand*3+3])
                if size:stacks[size]+=1
    return {'windows':len(packet['plans']),'empty_windows':sum(not any(p['counts']) for p in packet['plans']),
        'max_counts':[max(p['counts'][i] for p in packet['plans']) for i in range(3)],'events':events,'actions':len(packet['steps']),
        'hand_histories_absent_from_previous_global_event':absent,'actions_with_such_history':actions_absent,
        'hand_histories_carried_from_prior_window':carried,'literal_stack_sizes':dict(stacks)}


def run():
    source_run=json.loads((parent.OUT/'run.json').read_text());report=json.loads((parent.OUT/'report.json').read_text())
    audit=json.loads((parent.OUT/'artifact-audit.json').read_text())
    if report['identity']!=source_run['identity'] or audit['repeat_report_sha256']!=_sha(parent.OUT/'report.json'):raise ValueError('parent not audited')
    sources=source_run['config']['sources']
    if Counter(r['role'] for r in sources.values())!={'train':336,'validation':6} or 'fam:24227' in sources:raise ValueError('source roles changed')
    for item in sources.values():
        verify_sources(item['record'])
        if _sha(ROOT/item['payload'])!=item['payload_sha256']:raise ValueError('source payload changed')
    frozen=freeze_run(OUT,{'parent_identity':source_run['identity'],'source_inventory':sources,'code_sha256':_sha(__file__),'spec_sha256':_sha(SPEC),
        'failure_audit_sha256':_sha(failure.OUT/'report.json'),'parent_audit_sha256':_sha(parent.OUT/'artifact-audit.json'),
        'dependencies':{p:_sha(ROOT/p) for p in ('eval/joint_model.py','eval/joint_phrase.py')}})
    receipts={};stats=[]
    for family,item in sorted(sources.items()):
        original=torch.load(ROOT/item['payload'],weights_only=False)['source'];packet=encode(original)
        path=OUT/'packets'/(family.replace(':','_')+'.json')
        if path.exists():
            saved=json.loads(path.read_text())
            if saved!=packet:raise ValueError('packet differs from original source')
        else:_atomic(path,packet);saved=json.loads(path.read_text())
        restored=decode(saved)
        for key in ('notes','bombs','walls'):assert list(map(tuple,restored[key]))==list(map(tuple,original[key])),(family,key)
        for key in ('bpm','duration_beats','authored'):assert canonical(restored[key])==canonical(original[key]),(family,key)
        receipts[family]={'role':item['role'],'packet_sha256':_sha(path),'scene_sha256':packet['scene_sha256'],'source_payload_sha256':item['payload_sha256']}
        stats.append({'family':family,'role':item['role'],**coverage(packet)})
    counts={key:sum(r[key] for r in stats) for key in ('windows','empty_windows','events','actions','hand_histories_absent_from_previous_global_event',
        'actions_with_such_history','hand_histories_carried_from_prior_window')}
    result={'identity':frozen['identity'],'verified_source_roundtrips':len(stats),'totals':counts,
        'maximum_phrase_counts':[max(r['max_counts'][i] for r in stats) for i in range(3)],'families':stats,'receipts':receipts,
        'claim':'reference-budget reachability and causal history only; no serving predictor, new fit, generated candidate or quality gain'}
    path=OUT/'report.json'
    if path.exists():assert json.loads(path.read_text())==canonical(result)
    _atomic(path,result);return {k:v for k,v in result.items() if k not in ('families','receipts')}


if __name__=='__main__':print(json.dumps(run(),indent=1))
