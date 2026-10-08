"""Decompose frozen emitted rhythm and bound what the existing candidate pool can do."""
from collections import Counter
import json

import numpy as np

from eval import joint_diverse_fit as candidate,joint_aligned_selection as aligned,joint_uniform_audio as uniform,joint_ordered as ordered,joint_pilot as old
from eval.expressive_manifest import freeze_run
from eval.joint_phrase import ROOT,_atomic,_sha,encode_events
from eval.joint_model import source_rate
from eval.joint_export import read_chart
from eval.joint_deployment import reference_view
from eval.joint_continuity import describe
from qa.comparator_v2 import chart_verdict

OUT=ROOT/'experiments/joint-rhythm-failure-audit-v1'
PLAN=ROOT/'docs/specs/2026-09-30-joint-rhythm-failure-audit.md'
NAMES=['events_per_second','left_per_second','right_per_second','double_share','empty_window']


def selected_path(root,key,arm,name='ordered'):
    return old.PILOT/'development'/key/'b0' if arm['fallback'] else root/'development'/key/name/str(arm['selected_seed'])


def measure(path,record,item,scales):
    for name,sha in record['artifacts'].items():
        if _sha(path/name)!=sha:raise ValueError('frozen output changed')
    source=read_chart(path/'ExpertPlus.dat',path/'Info.dat');reference=item['data']['source'];audio=item['data']['audio']
    source['duration_beats']=reference['duration_beats']*source['bpm']/reference['bpm']
    aligned_source=reference_view(source,reference);profile=old.profile(aligned_source,audio);ref=old.profile(reference,audio)
    assert profile==record['profile'] and ref==record['reference_profile'];errors=old.errors(ref,profile,scales);assert errors==record['errors']
    assert old.errors(ref,ref,scales)['rhythm']['mean']==0
    a,b=np.asarray(ref['rhythm'],float),np.asarray(profile['rhythm'],float);scaled=np.abs(a-b)/np.array(scales['rhythm'])
    np.testing.assert_allclose(scaled.mean(0),errors['rhythm']['components'],atol=1e-12,rtol=0)
    assert abs(float(scaled.mean())-errors['rhythm']['mean'])<1e-12
    missed=(a[:,4]==0)&(b[:,4]==1);extra=(a[:,4]==1)&(b[:,4]==0)
    assert abs((int(missed.sum())+int(extra.sum()))/len(a)/scales['rhythm'][4]-errors['rhythm']['components'][4])<1e-12
    ev=encode_events(aligned_source['notes']);doubles=sum(all(e['hands']) for e in ev)
    worst=[]
    for index in sorted(range(len(a)),key=lambda i:(-float(scaled[i].mean()),i))[:5]:
        lo,hi=index*8,min((index+1)*8,reference['duration_beats']);i,j=np.searchsorted(audio['times'],np.array([lo,hi])*60/reference['bpm'])
        worst.append({'start_beat':lo,'end_beat':hi,'start_seconds':lo*60/reference['bpm'],
            'reference':a[index].tolist(),'generated':b[index].tolist(),'scaled_errors':dict(zip(NAMES,scaled[index].tolist())),
            'mean_error':float(scaled[index].mean()),'original_audio_mean_rms':float(np.mean(audio['rms'][i:j])) if j>i else None})
    return {'errors':errors,'rhythm_components':dict(zip(NAMES,errors['rhythm']['components'])),
        'component_contributions_to_mean':dict(zip(NAMES,(scaled.mean(0)/5).tolist())),
        'windows':len(a),'reference_empty_windows':int(a[:,4].sum()),'generated_empty_windows':int(b[:,4].sum()),
        'generated_rest_over_reference_notes':np.where(missed)[0].tolist(),'generated_notes_in_reference_rest':np.where(extra)[0].tolist(),
        'empty_component_from_missed_activity':int(missed.sum())/len(a)/scales['rhythm'][4],
        'empty_component_from_added_activity':int(extra.sum())/len(a)/scales['rhythm'][4],
        'requested_hand_event_rate':item['rate'],'reference_hand_event_rate':source_rate(reference),'achieved_hand_event_rate':source_rate(aligned_source),
        'events':len(ev),'notes':len(aligned_source['notes']),'doubles':doubles,'global_double_share':doubles/len(ev) if ev else None,
        'mean_window_double_share':float(b[:,3].mean()),'reference_mean_window_double_share':float(a[:,3].mean()),
        'worst_windows':worst,'continuity':describe(source)}


def run():
    report=json.loads((candidate.OUT/'report.json').read_text());audit=json.loads((candidate.OUT/'artifact-audit.json').read_text())
    if audit['repeat_report_sha256']!=_sha(candidate.OUT/'report.json'):raise ValueError('candidate report audit mismatch')
    scales=old.evaluation_freeze()['config']['scales'];inputs=ordered.inputs();work={};bound={}
    aligned_report=json.loads((aligned.OUT/'report.json').read_text());aligned_root=ROOT/aligned_report['inherited_result']['root'] if 'inherited_result' in aligned_report else aligned.OUT
    for family,item in inputs.items():
        key=family.replace(':','_');song=json.loads((candidate.OUT/'development'/key/'song.json').read_text())
        comparison=json.loads((aligned.OUT/'development'/key/'song.json').read_text()) if aligned_root==aligned.OUT else json.loads((aligned_root/'development'/key/'song.json').read_text())
        retrieval=json.loads((uniform.OUT/'development'/key/'song.json').read_text())
        arms={'b0':(old.PILOT/'development'/key/'b0',item['b0']),
            'original_retrieval':(selected_path(ordered.OUT,key,song['arms']['retrieval'],'retrieval'),song['arms']['retrieval']['selected']),
            'uniform_retrieval':(selected_path(uniform.OUT,key,retrieval['arms']['ordered']),retrieval['arms']['ordered']['selected']),
            'approved_model':(selected_path(aligned_root,key,comparison['arms']['ordered']),comparison['arms']['ordered']['selected']),
            'diverse_model':(selected_path(candidate.OUT,key,song['arms']['ordered']),song['arms']['ordered']['selected'])}
        candidates=[]
        for seed in range(6):
            path=candidate.OUT/'development'/key/'ordered'/str(seed);r=json.loads((path/'record.json').read_text())
            if r['ok']:
                assert r['machine']['verdict']==chart_verdict(r['machine']['contradictions'],r['machine']['support'])
                assert r['machine']['admitted']==(r['machine']['verdict']=='MACHINE_PASS_QUALITY_NOT_EVALUATED')
            candidates.append((path,r))
        work[family]=(item,song,arms,candidates)
        for path,record in list(arms.values())+candidates:
            for name in record['artifacts']:bound[str((path/name).relative_to(ROOT))]=_sha(path/name)
            if (path/'record.json').exists():bound[str((path/'record.json').relative_to(ROOT))]=_sha(path/'record.json')
        payload=old.OUT/'data'/(key+'.pt');receipt=json.loads(payload.with_suffix('.json').read_text())
        if _sha(payload)!=receipt['sha256']:raise ValueError('reference data changed')
        bound[str(payload.relative_to(ROOT))]=receipt['sha256']
        for root in (candidate.OUT,aligned_root,uniform.OUT):
            path=root/'development'/key/'song.json';bound[str(path.relative_to(ROOT))]=_sha(path)
    frozen=freeze_run(OUT,{'parent_identity':report['identity'],'code_sha256':_sha(__file__),'plan_sha256':_sha(PLAN),
        'reports':{str((root/'report.json').relative_to(ROOT)):_sha(root/'report.json') for root in (candidate.OUT,aligned.OUT,uniform.OUT,ordered.OUT)},
        'artifact_audit_sha256':_sha(candidate.OUT/'artifact-audit.json'),'scales':scales,'artifacts':bound,
        'dependencies':{p:_sha(ROOT/p) for p in ('eval/joint_pilot.py','eval/joint_export.py','eval/joint_deployment.py','eval/joint_continuity.py','qa/comparator_v2.py')}})
    families=[];verdicts=Counter()
    for family,(item,song,arms,candidates) in work.items():
        measured={name:measure(path,record,item,scales) for name,(path,record) in arms.items()};pool=[]
        for seed,(path,r) in enumerate(candidates):
            verdicts[r.get('machine',{}).get('verdict',r['reason'])]+=1
            value=measure(path,r,item,scales)['errors']['rhythm']['mean'] if r['ok'] else None
            pool.append({'seed':seed,'ok':r['ok'],'admitted':r.get('machine',{}).get('admitted',False),'rhythm_error':value})
        b0=measured['b0']['errors']['rhythm']['mean'];admitted=[r for r in pool if r['admitted']];complete=[r for r in pool if r['rhythm_error'] is not None]
        oracle={'admitted_only_or_forced_b0':min((r['rhythm_error'] for r in admitted),default=b0),
                'admitted_or_optional_b0':min([b0]+[r['rhythm_error'] for r in admitted]),
                'all_complete_or_optional_b0':min([b0]+[r['rhythm_error'] for r in complete]),
                'best_admitted_seed':min(admitted,key=lambda r:(r['rhythm_error'],r['seed']))['seed'] if admitted else None,
                'complete_range':[min(r['rhythm_error'] for r in complete),max(r['rhythm_error'] for r in complete)] if complete else None}
        families.append({'family':family,'fallback':song['arms']['ordered']['fallback'],'arms':measured,'pool':pool,'oracle':oracle})
    aggregate={}
    for arm in families[0]['arms']:
        aggregate[arm]={'mean_rhythm_error':float(np.mean([f['arms'][arm]['errors']['rhythm']['mean'] for f in families])),
            'component_mean_errors':{name:float(np.mean([f['arms'][arm]['rhythm_components'][name] for f in families])) for name in NAMES}}
    groups={}
    for fallback in (False,True):
        part=[f for f in families if f['fallback']==fallback]
        groups['fallback' if fallback else 'nonfallback']={'families':[f['family'] for f in part],
            'means':{arm:float(np.mean([f['arms'][arm]['errors']['rhythm']['mean'] for f in part])) if part else None for arm in aggregate},
            'contribution_to_diverse_mean':sum(f['arms']['diverse_model']['errors']['rhythm']['mean'] for f in part)/len(families)}
    limits={name:float(np.mean([f['oracle'][name] for f in families])) for name in ('admitted_only_or_forced_b0','admitted_or_optional_b0','all_complete_or_optional_b0')}
    target=.9*min(aggregate[a]['mean_rhythm_error'] for a in ('b0','original_retrieval'))
    result={'identity':frozen['identity'],'aggregate':aggregate,'fallback_decomposition':groups,'oracle_rhythm_lower_bounds':limits,
        'required_mean_rhythm_upper_bound':target,'any_pool_selector_can_meet_mean_rhythm_gate':limits['admitted_or_optional_b0']<=target,
        'even_unadmitted_pool_can_meet_mean_rhythm_gate':limits['all_complete_or_optional_b0']<=target,
        'attempt_verdicts':dict(verdicts),'families':families,
        'claim':'artifact-only diagnosis; oracle uses opened references and cannot serve; empty means no chart notes, not acoustic silence; old gate unchanged'}
    rp=OUT/'report.json'
    if rp.exists():assert json.loads(rp.read_text())==result
    _atomic(rp,result);return {k:v for k,v in result.items() if k!='families'}


if __name__=='__main__':print(json.dumps(run(),indent=1))
