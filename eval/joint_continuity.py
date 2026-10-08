"""Scoped directional/movement diagnostics, never a new quality or safety gate."""
import json
import math
from collections import Counter

import numpy as np
from parity import family
from eval.quality_metrics import unrounded_transitions
from eval.joint_phrase import ROOT,_atomic,_sha,encode_events,read_source
from eval.joint_export import read_chart
from eval import joint_pilot as old,joint_refine as typed,joint_likelihood as likelihood

OUT=ROOT/'experiments/joint-continuity-v1'


def describe(source):
    previous={};excluded=Counter();pairs=[]
    for event in encode_events(source['notes']):
        for hand,notes in enumerate(event['hands']):
            if not notes:continue
            heads=[n for n in notes if n[2]!=8]
            if len(heads)!=1:
                excluded['dot_only' if not heads else 'multi_head']+=1
                previous.pop(hand,None);continue
            c,l,d=heads[0];beat=event['beat'];now=(beat*60000/source['bpm'],hand,c,l,d)
            if hand in previous:
                old_beat,prev=previous[hand]
                transition=unrounded_transitions([prev,now])[0]
                gap=transition['ms'];before,after=family(prev[4]),family(d)
                category='recovery' if gap>=1000 else 'lateral_endpoint' if 'lateral' in (before,after) else 'fast' if gap<=250 else 'medium'
                pairs.append({**transition,'category':category,'same_family':before==after if category in ('fast','medium') else None,
                              'cross_window':int(old_beat//8)!=int(beat//8),'from_beat':old_beat,'to_beat':beat,
                              'from_direction':prev[4],'to_direction':d})
            previous[hand]=(beat,now)
    result={'head_groups':sum(len(h)>0 for e in encode_events(source['notes']) for h in e['hands']),
            'excluded_groups':dict(excluded),'transitions':len(pairs),'bins':{}}
    for name in ('fast','medium','lateral_endpoint','recovery'):
        rows=[p for p in pairs if p['category']==name];scoped=name in ('fast','medium')
        seams=[p for p in rows if p['cross_window']]
        result['bins'][name]={'opportunities':len(rows),'same_family':sum(p['same_family'] for p in rows) if scoped else None,
            'same_family_rate':sum(p['same_family'] for p in rows)/len(rows) if scoped and rows else None,
            'cross_window_opportunities':len(seams),'cross_window_same_family':sum(p['same_family'] for p in seams) if scoped else None,
            'movement_mean':float(np.mean([p['dist'] for p in rows])) if rows else None,
            'movement_p90':float(np.quantile([p['dist'] for p in rows],.9)) if rows else None,
            'speed_p90':float(np.quantile([p['speed'] for p in rows],.9)) if rows else None}
    result['fast_same_family_examples']=[p for p in pairs if p['category']=='fast' and p['same_family']][:12]
    result['worst_movement_speed']=sorted((p for p in pairs if p['category']!='recovery'),key=lambda p:-p['speed'])[:5]
    return result


def controls():
    source={'bpm':120.,'duration_beats':8.,'bombs':[],'walls':[],'notes':[]}
    for i in range(4):
        for hand,col,layer in ((0,0,i%2),(1,3,i//2)):
            source['notes'].append((1+i*.25,hand,col,layer,i%2))
    repeated={**source,'notes':[(b,h,c,l,0) for b,h,c,l,d in source['notes']]}
    audio={'times':np.arange(0,4,.01),'rms':np.ones(400),'onset':np.ones(400)}
    a,b=old.profile(source,audio),old.profile(repeated,audio)
    x,y=describe(source),describe(repeated)
    assert a==b and x['bins']['fast']['same_family']==0 and y['bins']['fast']['same_family']==6
    return {'profiles_identical':True,'alternating':x,'repeated':y,'sources':[source,repeated],
            'interpretation':'directional sequence differs despite equal profiles; same-family heuristic is not a safety proof'}


def audit():
    from eval.expressive_manifest import freeze_run
    parent=likelihood.freeze_experiment();manifest=json.loads((old.OUT/'run.json').read_text())
    sources={};bindings={}
    for record in manifest['config']['records']:
        if record['fit_role']!='development':continue
        fam=record['family'];key=fam.replace(':','_');root=old.PILOT/'development'/key
        bindings.update({s['path']:s['sha256'] for s in record['sources'].values()})
        sources[fam]={'human':read_source(record)}
        original=json.loads((root/'song.json').read_text());rseed=original['arms']['retrieval']['selected_seed']
        retrieval=root/'retrieval'/str(rseed) if rseed is not None else root/'b0'
        v3root=typed.start.OUT/'development'/key;v3=json.loads((v3root/'song.json').read_text())
        v3seed=v3['arms']['model']['selected_seed']
        v4root=typed.OUT/'development'/key;v4=json.loads((v4root/'song.json').read_text())
        v6=json.loads((likelihood.OUT/'development'/key/'song.json').read_text())
        paths={'b0':root/'b0','retrieval':retrieval,'v3':v3root/'model'/str(v3seed) if v3seed is not None else root/'b0',
               'v4':v4root/'refined'/str(v4['selected_seed']) if v4['selected_seed'] is not None else retrieval,
               'v6':v4root/'refined'/str(v6['selected_seed']) if v6['selected_seed'] is not None else retrieval}
        for arm,path in paths.items():
            sources[fam][arm]=read_chart(path/'ExpertPlus.dat',path/'Info.dat')
            for name in ('ExpertPlus.dat','Info.dat'):bindings[str((path/name).relative_to(ROOT))]=_sha(path/name)
        for path in (root/'song.json',v3root/'song.json',v4root/'song.json',likelihood.OUT/'development'/key/'song.json'):
            bindings[str(path.relative_to(ROOT))]=_sha(path)
    frozen=freeze_run(OUT,{'parent_identity':parent['identity'],'code_sha256':_sha(__file__),'inputs':bindings,
        'dependencies':{p:_sha(ROOT/p) for p in ('parity.py','motion.py','eval/quality_metrics.py','eval/joint_export.py','eval/joint_phrase.py')},
        'scope':'descriptive single-head directional chains; not admission, no threshold fit','controls':controls()})
    rows={fam:{arm:describe(src) for arm,src in arms.items()} for fam,arms in sources.items()}
    aggregate={}
    for arm in ('human','b0','retrieval','v3','v4','v6'):
        aggregate[arm]={}
        for category in ('fast','medium'):
            entries=[arms[arm]['bins'][category] for arms in rows.values()]
            total=sum(x['opportunities'] for x in entries);same=sum(x['same_family'] for x in entries)
            rates=[x['same_family_rate'] for x in entries if x['same_family_rate'] is not None]
            seams=sum(x['cross_window_opportunities'] for x in entries);badseams=sum(x['cross_window_same_family'] for x in entries)
            aggregate[arm][category]={'opportunities':total,'same_family':same,'pooled_rate':same/total if total else None,
                'family_mean_rate':float(np.mean(rates)) if rates else None,'families_with_opportunities':len(rates),
                'cross_window_opportunities':seams,'cross_window_same_family':badseams,
                'cross_window_pooled_rate':badseams/seams if seams else None}
    result={'identity':frozen['identity'],'status':'DESCRIPTIVE_AUDIT_COMPLETE','aggregate':aggregate,'families':rows,
            'claim':'scoped directional and movement diagnostics; no universal flow/safety label or new acceptance gate'}
    _atomic(OUT/'report.json',result);return result


if __name__=='__main__':print(json.dumps(audit(),indent=1))
