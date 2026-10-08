"""Exact origins of a frozen arc proxy, not physical hand trajectories."""
from collections import Counter
import json
import numpy as np
import torch

from eval import phrase_native as parent,phrase_native_likelihood as ranked,joint_arc_audit as dots,expression_profile as expression
from eval import joint_ordered as ordered,joint_pilot as old,joint_uniform_audio as uniform
from eval.expressive_manifest import freeze_run
from eval.joint_deployment import reference_view
from eval.joint_export import read_chart
from eval.joint_phrase import ROOT,_sha,_atomic,phrase_windows

OUT=ROOT/'experiments/phrase-native-arc-origin-v1'
PLAN=ROOT/'docs/specs/2026-09-30-budget-native-arc-origin.md'
MODES=('legacy','dot_split','unambiguous','unambiguous_without_seams')


def window(notes,bpm,origins=None):
    heads=[tuple(n) for n in notes if n[4]!=8];events=Counter((n[0],n[1]) for n in heads);runs=[]
    t0=heads[0][0]*60/bpm if heads else 0.
    for hand in (0,1):
        seq=[n for n in heads if n[1]==hand]
        for i in range(len(seq)-3):
            quad=seq[i:i+4];numeric=[(b*60/bpm-t0,h,c,l,d) for b,h,c,l,d in quad]
            if not expression._arcs(numeric)['n_runs']:continue
            dot=any(n[1]==hand and n[4]==8 and quad[0][0]<=n[0]<=quad[-1][0] for n in notes)
            stack=any(h==hand and quad[0][0]<=b<=quad[-1][0] and count>1 for (b,h),count in events.items())
            donors=[origins[n] for n in quad] if origins is not None else None
            runs.append({'notes':quad,'same_hand_dot':dot,'simultaneous_multi_head':stack,
                         'donor_indices':donors,'crosses_donors':len(set(donors))>1 if donors is not None else False})
    prior=dots.window_ledger([(h,c,l,d) for b,h,c,l,d in notes],[b*60/bpm for b,h,c,l,d in notes])
    assert len(runs)==prior['legacy_runs'] and sum(r['same_hand_dot'] for r in runs)==prior['dot_bridged_runs']
    counts={'legacy':len(runs),'dot_split':sum(not r['same_hand_dot'] for r in runs),
        'unambiguous':sum(not r['same_hand_dot'] and not r['simultaneous_multi_head'] for r in runs),
        'unambiguous_without_seams':sum(not r['same_hand_dot'] and not r['simultaneous_multi_head'] and not r['crosses_donors'] for r in runs)}
    return {'supported':prior['supported'],'heads':len(heads),'counts':counts,'runs':runs,
            'values':{k:100*v/len(heads) if prior['supported'] else None for k,v in counts.items()}}


def describe(source,origins=None):
    rows=[]
    for w in phrase_windows(source['notes'],source['bombs'],source['walls'],source['duration_beats']):
        notes=[(e['beat'],h,c,l,d) for e in w['events'] for h,hs in enumerate(e['hands']) for c,l,d in hs]
        rows.append({'start':w['start'],**window(notes,source['bpm'],origins)})
    runs=[r for w in rows for r in w['runs']]
    return {'windows':rows,'totals':{'legacy':len(runs),'crosses_donors':sum(r['crosses_donors'] for r in runs),
        'same_hand_dot':sum(r['same_hand_dot'] for r in runs),'simultaneous_multi_head':sum(r['simultaneous_multi_head'] for r in runs),
        'unambiguous_internal':sum(not(r['same_hand_dot'] or r['simultaneous_multi_head'] or r['crosses_donors']) for r in runs)}}


def distances(ref,got,scale):
    result={}
    for mode in MODES:
        pairs=[(a['values'][mode],b['values'][mode]) for a,b in zip(ref['windows'],got['windows']) if a['supported'] and b['supported']]
        result[mode]=float(np.mean([abs(a-b)/scale for a,b in pairs])) if pairs else None
    return result


def run():
    native=parent.freeze();selection=ranked.freeze();audit=json.loads((ranked.OUT/'artifact-audit.json').read_text())
    if not audit['repeat_verified'] or audit['report_sha256']!=_sha(ranked.OUT/'report.json'):raise ValueError('ranked result not repeated')
    frozen=freeze_run(OUT,{'parent_identity':native['identity'],'ranked_identity':selection['identity'],
        'parent_audit_sha256':_sha(parent.OUT/'artifact-audit.json'),'ranked_audit_sha256':_sha(ranked.OUT/'artifact-audit.json'),
        'code_sha256':_sha(__file__),'plan_sha256':_sha(PLAN),
        'control_exports':{str(p.relative_to(ROOT)):_sha(p) for base in (old.PILOT/'development',ordered.OUT/'development')
            for p in base.rglob('*') if p.is_file() and p.name in ('ExpertPlus.dat','Info.dat')},
        'dependencies':{p:_sha(ROOT/p) for p in ('eval/joint_arc_audit.py','eval/expression_profile.py','eval/joint_phrase.py','eval/joint_deployment.py')}})
    bank=torch.load(uniform.OUT/'retrieval.pt',weights_only=False);lookup={(e['family'],e['start']):e for e in bank['entries']}
    scales=old.evaluation_freeze()['config']['scales'];families=[]
    for family,item in ordered.inputs().items():
        key=family.replace(':','_');root=parent.OUT/'development'/key;song=json.loads((root/'song.json').read_text())
        ranked_song=json.loads((ranked.OUT/'development'/key/'song.json').read_text());reference=describe(item['data']['source']);attempts=[]
        for seed,r in enumerate(song['arms']['ordered']['attempts']):
            if not r['ok']:attempts.append(None);continue
            source=read_chart(root/'ordered'/str(seed)/'ExpertPlus.dat',root/'ordered'/str(seed)/'Info.dat');source['duration_beats']=item['source']['duration_beats']
            origins={}
            for i,donor in enumerate(r['donors']):
                for b,h,c,l,d in lookup[tuple(donor)]['notes']:
                    if b+i*8<source['duration_beats']:origins[(b+i*8,h,c,l,d)]=i
            assert sorted(origins)==sorted(map(tuple,source['notes']))
            view=reference_view(source,item['data']['source']);view_origins={tuple(b):origins[tuple(a)] for a,b in zip(source['notes'],view['notes'])}
            description=describe(view,view_origins);distance=distances(reference,description,scales['geometry'][1])
            assert np.isclose(distance['legacy'],r['errors']['geometry']['components'][1],rtol=0,atol=1e-10)
            attempts.append({'seed':seed,'description':description,'arc_distances':distance})
        arms={}
        for name,path,record in [('b0',old.PILOT/'development'/key/'b0',song['b0']),
            ('retrieval',ordered.OUT/'development'/key/'retrieval'/str(song['arms']['retrieval']['selected_seed']),song['arms']['retrieval']['selected'])]:
            if name=='retrieval' and song['arms']['retrieval']['fallback']:path=old.PILOT/'development'/key/'b0'
            for filename in ('ExpertPlus.dat','Info.dat'):assert _sha(path/filename)==record['artifacts'][filename]
            source=read_chart(path/'ExpertPlus.dat',path/'Info.dat')
            source['duration_beats']=item['data']['source']['duration_beats']*source['bpm']/item['data']['source']['bpm']
            description=describe(reference_view(source,item['data']['source']));distance=distances(reference,description,scales['geometry'][1])
            assert np.isclose(distance['legacy'],record['errors']['geometry']['components'][1],rtol=0,atol=1e-10)
            arms[name]={'description':description,'arc_distances':distance}
        for name,selected in [('first_admitted',song['arms']['ordered']),('likelihood',ranked_song['arms']['ordered'])]:
            arms[name]=arms['b0'] if selected['fallback'] else attempts[selected['selected_seed']]
        families.append({'family':family,'reference':reference,'attempts':attempts,'arms':arms})
    aggregate={name:{'arc_distances':{mode:float(np.mean([f['arms'][name]['arc_distances'][mode] for f in families])) for mode in MODES},
        'totals':{key:sum(f['arms'][name]['description']['totals'][key] for f in families) for key in families[0]['reference']['totals']}}
        for name in families[0]['arms']}
    result={'identity':frozen['identity'],'families':families,'aggregate':aggregate,
        'claim':'arc-proxy origin and counterfactual descriptor distances only; no repaired charts, physical flow claim or changed gate'}
    result=json.loads(json.dumps(result));path=OUT/'report.json'
    if path.exists():assert json.loads(path.read_text())==result
    _atomic(path,result);return {k:v for k,v in result.items() if k!='families'}


if __name__=='__main__':print(json.dumps(run(),indent=1))
