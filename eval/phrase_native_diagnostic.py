"""Separate budget prediction, native projection and fallback costs after generation."""
import json
import numpy as np

from eval import phrase_native as candidate,phrase_budget as planner,joint_uniform_audio as control,joint_ordered as ordered,joint_pilot as old
from eval.expressive_manifest import freeze_run
from eval.joint_deployment import reference_view
from eval.joint_export import read_chart
from eval.joint_phrase import ROOT,_sha,_atomic

OUT=ROOT/'experiments/phrase-native-diagnostic-v1'


def run():
    parent=candidate.freeze();audit=json.loads((candidate.OUT/'artifact-audit.json').read_text())
    if audit['repeat_report_sha256']!=_sha(candidate.OUT/'report.json'):raise ValueError('candidate not audited')
    bound={};work={}
    for family,item in ordered.inputs().items():
        root=candidate.OUT/'development'/family.replace(':','_');song=json.loads((root/'song.json').read_text())
        bound[str((root/'song.json').relative_to(ROOT))]=_sha(root/'song.json');records=[]
        for seed in range(6):
            path=root/'ordered'/str(seed);r=old.cached_record(path/'record.json',parent['identity']);assert r is not None
            records.append(r);bound[str((path/'record.json').relative_to(ROOT))]=_sha(path/'record.json')
            for name in r['artifacts']:bound[str((path/name).relative_to(ROOT))]=_sha(path/name)
        work[family]=(item,song,records)
    frozen=freeze_run(OUT,{'parent_identity':parent['identity'],'parent_report_sha256':_sha(candidate.OUT/'report.json'),
        'parent_audit_sha256':_sha(candidate.OUT/'artifact-audit.json'),'code_sha256':_sha(__file__),'artifacts':bound,
        'dependencies':{p:_sha(ROOT/p) for p in ('eval/phrase_budget.py','eval/joint_pilot.py','eval/joint_deployment.py')}})
    scales=old.evaluation_freeze()['config']['scales'];rows=[]
    for family,(item,song,records) in work.items():
        # This reference-derived view is opened only after generation and auditing.
        reference=reference_view(item['data']['source'],item['source'])
        ref=old.profile(reference,item['data']['audio'])['rhythm'];plans=records[0]['plans'];pred=planner.rhythm(plans,item['source']['bpm'])
        predicted_error=planner.error(ref,pred,scales);candidates=[]
        for seed,r in enumerate(records):
            if not r['ok']:continue
            realized=[{**p,'counts':p['realized_counts']} for p in r['budget_ledger']]
            profile=planner.rhythm(realized,item['source']['bpm'])
            path=candidate.OUT/'development'/family.replace(':','_')/'ordered'/str(seed)
            source=read_chart(path/'ExpertPlus.dat',path/'Info.dat');source['duration_beats']=item['source']['duration_beats']
            np.testing.assert_allclose(profile,old.profile(source,item['data']['audio'])['rhythm'],rtol=0,atol=1e-12)
            candidates.append({'seed':seed,'admitted':r['machine']['admitted'],
                'serving_grid_error':planner.error(ref,profile,scales),'projection_distance_from_plan':planner.error(pred,profile,scales),
                'original_gate_rhythm':r['errors']['rhythm']['mean'],'original_gate_geometry':r['errors']['geometry']['mean'],
                'requested_rate_ratio':sum(a+b+2*c for a,b,c in (p['counts'] for p in realized))/(item['rate']*item['source']['duration_beats']*60/item['source']['bpm'])})
        selected=song['arms']['ordered'];b0=song['b0'];admitted=[c for c in candidates if c['admitted']]
        oracles={}
        for axis in ('rhythm','geometry'):
            key='original_gate_'+axis;base=b0['errors'][axis]['mean']
            oracles[axis]={'admitted_or_b0':min([base]+[c[key] for c in admitted]),
                           'all_complete_or_b0':min([base]+[c[key] for c in candidates])}
        requested=sum(a+b+2*c for a,b,c in (p['counts'] for p in plans))/(item['rate']*item['source']['duration_beats']*60/item['source']['bpm'])
        rows.append({'family':family,'serving_bpm':item['source']['bpm'],'authored_bpm':item['data']['source']['bpm'],
            'prediction_error_on_serving_grid':predicted_error,'predicted_workload_ratio':requested,
            'predicted_notes_in_reference_rests':[i for i,(a,b) in enumerate(zip(ref,pred)) if a[4] and not b[4]],
            'predicted_rests_over_reference_notes':[i for i,(a,b) in enumerate(zip(ref,pred)) if b[4] and not a[4]],
            'fallback':selected['fallback'],'selected_seed':selected['selected_seed'],
            'selected_rhythm_error':selected['selected']['errors']['rhythm'],'selected_geometry_error':selected['selected']['errors']['geometry'],
            'candidates':candidates,'oracle_bounds_on_opened_reference':oracles})
    aggregate={'serving_grid_prediction_mean':float(np.mean([r['prediction_error_on_serving_grid']['mean'] for r in rows])),
        'serving_grid_prediction_components':np.mean([r['prediction_error_on_serving_grid']['components'] for r in rows],axis=0).tolist(),
        'serving_grid_mean_rendered':float(np.mean([np.mean([c['serving_grid_error']['mean'] for c in r['candidates']]) for r in rows if r['candidates']])),
        'oracle_bounds':{axis:{kind:float(np.mean([r['oracle_bounds_on_opened_reference'][axis][kind] for r in rows]))
            for kind in ('admitted_or_b0','all_complete_or_b0')} for axis in ('rhythm','geometry')}}
    result={'identity':frozen['identity'],'families':rows,'aggregate':aggregate,
        'claim':'serving-grid decomposition is diagnostic only; original gate remains on authored scoring windows; oracle references never select serving output'}
    path=OUT/'report.json'
    if path.exists():assert json.loads(path.read_text())==result
    _atomic(path,result);return {k:v for k,v in result.items() if k!='families'}


if __name__=='__main__':print(json.dumps(run(),indent=1))
