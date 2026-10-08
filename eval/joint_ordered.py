"""One ordered-audio retrieval comparison with deployment-derived timing."""
import json
import time
from pathlib import Path

import numpy as np
import torch

from eval import joint_pilot as old,joint_likelihood as likelihood
from eval.joint_model import _torch_save,source_rate
from eval.joint_decode import JointState
from eval.joint_deployment import deployment_input,reference_view
from eval.joint_export import export_chart,assert_same
from eval.joint_phrase import ROOT,_atomic,_sha,encode_events

OUT=ROOT/'experiments/joint-phrase-v7/ordered-retrieval'
SPEC=ROOT/'docs/specs/2026-09-30-ordered-retrieval-deployment-design.md'


def temporal_descriptor(audio,start,end,bpm):
    result=np.zeros((16,2),dtype=np.float32)
    for i in range(16):
        lo=start+i*.5;hi=min(lo+.5,end)
        if hi<=lo:continue
        a,b=np.searchsorted(audio['times'],np.array([lo,hi])*60/bpm)
        if b>a:result[i]=audio['x'][a:b,-2:].mean(0)
    return result.reshape(-1)


def ordered_bank(train,base):
    if any(d['fit_role']!='train' for d in train):raise ValueError('train-only phrase bank')
    lookup={d['family']:d for d in train}
    if len(lookup)!=len(train) or set(lookup)!={e['family'] for e in base['entries']}:
        raise ValueError('bank family inventory mismatch')
    features=[]
    for e in base['entries']:
        d=lookup[e['family']]
        features.append(temporal_descriptor(d['audio'],e['start'],e['start']+8,d['source']['bpm']))
    x=np.stack(features);mean=x.mean(0);sd=np.maximum(x.std(0),1e-6)
    return {**base,'temporal_mean':mean,'temporal_sd':sd,'temporal_x':(x-mean)/sd}


def retrieve(source,audio,bank,rate,seed=0):
    """The original literal retrieval loop; only its distance gains temporal order."""
    state=JointState(source['duration_beats']);used=[]
    for start in np.arange(0.,source['duration_beats'],8.):
        end=min(start+8,source['duration_beats'])
        q=(old.phrase_descriptor(audio,start,end,source['bpm'],rate)-bank['mean'])/bank['sd']
        t=(temporal_descriptor(audio,start,end,source['bpm'])-bank['temporal_mean'])/bank['temporal_sd']
        distances=np.square(bank['x']-q).mean(1)+np.square(bank['temporal_x']-t).mean(1)
        order=np.argsort(distances,kind='stable')[:32];compatible=[];previous=state.notes()
        for ix in order:
            entry=bank['entries'][ix];key=(entry['family'],entry['start'])
            if used and key==used[-1]:continue
            notes=[(b+float(start),h,c,l,d) for b,h,c,l,d in entry['notes'] if b<end-start]
            if old._join_ok(previous,notes,source['bpm']):
                compatible.append((key,notes))
                if len(compatible)==6:break
        if not compatible:
            return {'ok':False,'reason':'no_compatible_retrieval_join','beat':float(start),'donors':used}
        key,notes=compatible[seed%len(compatible)]
        for event in encode_events(notes):state.append(event)
        state.rest(float(end));used.append(key)
    return {'ok':bool(state.events),'reason':None if state.events else 'empty_chart',
            'source':{**source,'notes':state.notes(),'events':state.events},
            'donors':used,'unique_donors':len(set(used)),'attempt_seed':seed}


def inputs():
    data=old.load_data('development')
    manifest=json.loads((old.OUT/'run.json').read_text())
    audio_paths={r['family']:Path(r['sources']['audio']['path']) for r in manifest['config']['records']}
    result={}
    for d in data:
        key=d['family'].replace(':','_');base=ROOT/'experiments/expressive-v1/fresh'
        source,rate,proof=deployment_input(audio_paths[d['family']],base/'mi'/key/'gen.osu',base/'partial'/key/'b0.json')
        original=json.loads((old.PILOT/'development'/key/'song.json').read_text())
        result[d['family']]={'data':d,'source':source,'rate':rate,'proof':proof,'b0':original['b0']}
    if len(result)!=8:raise ValueError('development inventory changed')
    return result


def freeze_experiment():
    from eval.expressive_manifest import freeze_run
    parent=likelihood.freeze_experiment();audit=ROOT/'experiments/joint-phrase-v5/deployment-input-audit-native-duration'
    ar=json.loads((audit/'run.json').read_text())
    if ar['config']['code_sha256']!=_sha(ROOT/'eval/joint_deployment.py'):raise ValueError('deployment adapter changed')
    base=old.evaluation_freeze();receipt=json.loads((old.PILOT/'retrieval-receipt.json').read_text())
    if receipt['identity']!=base['identity'] or receipt['sha256']!=_sha(old.PILOT/'retrieval.pt'):
        raise ValueError('original retrieval bank changed')
    return freeze_run(OUT,{'parent_identity':parent['identity'],'spec_sha256':_sha(SPEC),
             'code_sha256':_sha(__file__),'v6_report_sha256':_sha(likelihood.OUT/'report.json'),
             'input_audit_identity':ar['identity'],'input_audit_report_sha256':_sha(audit/'report.json'),
             'deployment_code_sha256':_sha(ROOT/'eval/joint_deployment.py'),
             'inputs':{fam:item['proof'] for fam,item in inputs().items()},
             'original_bank_receipt':receipt,'bins':16,'bin_beats':.5,'temporal_group_weight':1.})


def prepare():
    f=freeze_experiment();path=OUT/'retrieval.pt';receipt_path=OUT/'retrieval-receipt.json'
    if not receipt_path.exists():
        base=torch.load(old.PILOT/'retrieval.pt',weights_only=False)
        train=old.load_data('train');bank=ordered_bank(train,base);del train
        _torch_save(path,bank);_atomic(receipt_path,{'identity':f['identity'],'sha256':_sha(path),
                                                  'entries':len(bank['entries'])})
    rec=json.loads(receipt_path.read_text())
    if rec['identity']!=f['identity'] or rec['sha256']!=_sha(path):raise ValueError('ordered bank changed')
    return f,rec


def measure(attempt,data,dest,identity,scales,kit,prior_path=None):
    source=attempt.pop('source',None)
    rec={**attempt,'family':data['family'],'identity':identity,'artifacts':{}}
    if not attempt['ok']:
        _atomic(dest/'record.json',rec);return rec
    if source is None or not source['notes']:raise ValueError('successful empty retrieval')
    emitted=export_chart(source,dest,njs=18);assert_same(source,emitted,18,0)
    rec['artifacts']={name:_sha(dest/name) for name in ('ExpertPlus.dat','Info.dat')}
    view=reference_view({**source,**emitted},data['source'])
    rec['profile']=old.profile(view,data['audio']);rec['reference_profile']=old.profile(data['source'],data['audio'])
    rec['errors']=old.errors(rec['reference_profile'],rec['profile'],scales)
    rec.update(serving_bpm=source['bpm'],measurement_bpm=view['bpm'],emitted_notes=len(source['notes']))
    prior=json.loads(prior_path.read_text()) if prior_path is not None and prior_path.exists() else None
    if prior and prior.get('machine') and prior.get('artifacts')==rec['artifacts']:
        # Exact chart + Info identity is required; similar notes do not reuse QA.
        rec['machine']=prior['machine'];rec['qa_reused_from']=str(prior_path.relative_to(ROOT))
        rec['qa_reused_record_sha256']=_sha(prior_path)
    else:rec['machine']=old.machine_measure(emitted,kit)
    _atomic(dest/'record.json',rec)
    return rec


def select(records,baseline):
    seed=next((i for i,r in enumerate(records) if r['ok'] and r.get('machine',{}).get('admitted')),None)
    return {'selected_seed':seed,'fallback':seed is None,'selected':baseline if seed is None else records[seed],
            'attempts':records}


def decide(songs,frozen):
    aggregate={};arms=('b0','retrieval','ordered')
    def chosen(song,arm):return song['b0'] if arm=='b0' else song['arms'][arm]['selected']
    for arm in arms:
        aggregate[arm]={}
        for axis in ('rhythm','geometry'):
            es=[chosen(s,arm)['errors'][axis] for s in songs]
            aggregate[arm][axis]={'mean':float(np.mean([e['mean'] for e in es])) if all(e['mean'] is not None for e in es) else None,
                'components':[float(np.mean(v)) if all(x is not None for x in v) else None for v in zip(*(e['components'] for e in es))],
                'supported_windows':np.sum([e['supported_windows'] for e in es],axis=0).tolist()}
    gates={'complete_panel':len(songs)==8,'six_nonfallback':sum(not s['arms']['ordered']['fallback'] for s in songs)>=6}
    rng=np.random.default_rng(20260930);ix=rng.integers(0,len(songs),(2000,len(songs)));intervals={}
    for other in ('b0','retrieval'):
        for axis in ('rhythm','geometry'):
            a,b=aggregate['ordered'][axis],aggregate[other][axis]
            gates[f'{axis}_10percent_vs_{other}']=a['mean'] is not None and b['mean'] is not None and a['mean']<=.9*b['mean'] and a['mean']<b['mean']
            gates[f'{axis}_components_vs_{other}']=all(x is not None and y is not None and x<=1.1*y+.01 for x,y in zip(a['components'],b['components']))
            gates[f'{axis}_coverage_vs_{other}']=all(x>=y for x,y in zip(a['supported_windows'],b['supported_windows']))
            pairs=[(chosen(s,'ordered')['errors'][axis]['mean'],chosen(s,other)['errors'][axis]['mean']) for s in songs]
            intervals[f'{axis}_ordered_minus_{other}']=np.quantile(np.array([a-b for a,b in pairs])[ix].mean(1),[.025,.975]).tolist() if all(a is not None and b is not None for a,b in pairs) else None
        gates['support_vs_'+other]=all(chosen(s,'ordered')['machine']['support']['share_supported'] is not None and
             chosen(s,other)['machine']['support']['share_supported'] is not None and
             chosen(s,'ordered')['machine']['support']['share_supported']>=chosen(s,other)['machine']['support']['share_supported']-.05 for s in songs)
    gates['no_new_contradictions']=all(chosen(s,'ordered')['machine']['contradictions']['status'] not in
             ('STRUCTURAL_CONTRADICTION','MODEL_CONTRADICTION') or chosen(s,'ordered')['machine']['contradictions']==s['b0']['machine']['contradictions'] for s in songs)
    report={'identity':frozen['identity'],'status':'DEVELOPMENT_POSITIVE' if all(gates.values()) else 'DEVELOPMENT_NEGATIVE',
            'release_eligible':False,'gates':gates,'aggregate':aggregate,'bootstrap_95_difference':intervals,
            'panel':json.loads((old.PILOT/'report.json').read_text())['panel'],
            'families':[{'family':s['family'],'serving_bpm':s['serving_bpm'],'requested_rate':s['requested_rate'],
                'reference_rate':s['reference_rate'],'audio':{a:chosen(s,a)['profile']['audio'] for a in arms},
                'arms':{a:{'selected_seed':v['selected_seed'],'fallback':v['fallback'],'attempts':[
                    {k:r.get(k) for k in ('ok','reason','elapsed_s','unique_donors','qa_reused_from')}|{'verdict':r.get('machine',{}).get('verdict')}
                    for r in v['attempts']]} for a,v in s['arms'].items()}} for s in songs],
            'budget':'six original and six ordered retrieval candidates per family; no geometry refinements or fitting',
            'claim':'opened development comparison with deployment timing, not fresh confirmation or universal map quality'}
    _atomic(OUT/'report.json',report);return report


def run(deadline_seconds=2700):
    deadline=time.monotonic()+min(deadline_seconds,2700);torch.set_num_threads(4)
    frozen,receipt=prepare();bank=torch.load(OUT/'retrieval.pt',weights_only=False)
    scales=old.evaluation_freeze()['config']['scales'];kit=old.machine_tools();songs=[]
    for family,item in inputs().items():
        d=item['data'];key=family.replace(':','_');dest=OUT/'development'/key
        arms={}
        for arm,generate in (('retrieval',old.retrieve_song),('ordered',retrieve)):
            records=[]
            for seed in range(6):
                target=dest/arm/str(seed);rec=old.cached_record(target/'record.json',frozen['identity'])
                if rec is None:
                    if time.monotonic()>=deadline:return {'status':'INCOMPLETE','families':len(songs),'family':family,'arm':arm,'next_seed':seed}
                    started=time.monotonic();attempt=generate(item['source'],d['audio'],bank,item['rate'],seed)
                    attempt['elapsed_s']=time.monotonic()-started
                    prior=old.PILOT/'development'/key/'retrieval'/str(seed)/'record.json' if arm=='retrieval' else None
                    rec=measure(attempt,d,target,frozen['identity'],scales,kit,prior)
                    print(f'{arm} {family} s{seed}: {rec["ok"]} {rec.get("machine",{}).get("verdict")}',flush=True)
                records.append(rec)
            arms[arm]=select(records,item['b0'])
        song={'identity':frozen['identity'],'family':family,'b0':item['b0'],'arms':arms,
              'requested_rate':item['rate'],'reference_rate':source_rate(d['source']),'serving_bpm':item['source']['bpm']}
        _atomic(dest/'song.json',song);songs.append(song)
    prepare();return decide(songs,frozen)


if __name__=='__main__':
    print(json.dumps(run(),indent=1))
