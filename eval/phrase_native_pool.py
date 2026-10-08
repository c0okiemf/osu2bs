"""Reference-opened feasibility bound, explicitly not a serving selector."""
import json
import numpy as np
from scipy.optimize import Bounds,LinearConstraint,milp

from eval import phrase_native as parent,joint_ordered as ordered
from eval.expressive_manifest import freeze_run
from eval.joint_phrase import ROOT,_sha,_atomic

OUT=ROOT/'experiments/phrase-native-pool-feasibility-v1'
PLAN=ROOT/'docs/specs/2026-09-30-budget-native-pool-feasibility.md'


def allowed(record,b0,retrieval):
    if not record['ok'] or not record['machine']['admitted']:return False
    machine=record['machine'];share=machine['support']['share_supported']
    if share is None or any(c['machine']['support']['share_supported'] is None or share<c['machine']['support']['share_supported']-.05 for c in (b0,retrieval)):return False
    if machine['contradictions']['status'] in ('STRUCTURAL_CONTRADICTION','MODEL_CONTRADICTION') and machine['contradictions']!=b0['machine']['contradictions']:return False
    return all(record['errors'][axis]['mean'] is not None and all(x is not None for x in record['errors'][axis]['components']) for axis in ('rhythm','geometry'))


def run():
    frozen_parent=parent.freeze();report=json.loads((parent.OUT/'report.json').read_text());audit=json.loads((parent.OUT/'artifact-audit.json').read_text())
    assert audit['repeat_report_sha256']==_sha(parent.OUT/'report.json')
    songs=[json.loads(p.read_text()) for p in sorted((parent.OUT/'development').glob('*/song.json'))]
    assert len(songs)==8
    records={str(p.relative_to(ROOT)):_sha(p) for p in (parent.OUT/'development').rglob('*.json')}
    frozen=freeze_run(OUT,{'parent_identity':frozen_parent['identity'],'parent_audit_sha256':_sha(parent.OUT/'artifact-audit.json'),
        'parent_report_sha256':_sha(parent.OUT/'report.json'),'records':records,'code_sha256':_sha(__file__),
        'plan_sha256':_sha(PLAN),'gate_code_sha256':_sha(ordered.__file__)})
    options=[]
    for i,song in enumerate(songs):
        for seed,r in [(None,song['b0'])]+list(enumerate(song['arms']['ordered']['attempts'])):
            if allowed(r,song['b0'],song['arms']['retrieval']['selected']):options.append({'family_index':i,'seed':seed,'record':r})
    n=len(options);rows=[];lower=[];upper=[];names=[]
    def constraint(name,values,lo,hi):
        names.append(name);rows.append(values);lower.append(lo);upper.append(hi)
    for i,song in enumerate(songs):constraint('one:'+song['family'],[float(o['family_index']==i) for o in options],1.,1.)
    constraint('six_nonfallback',[float(o['seed'] is not None) for o in options],6.,np.inf)
    for arm in ('b0','retrieval'):
        for axis in ('rhythm','geometry'):
            control=report['aggregate'][arm][axis]
            constraint(axis+'_10percent_vs_'+arm,[o['record']['errors'][axis]['mean']/8 for o in options],-np.inf,.9*control['mean'])
            for k in range(5):
                constraint(axis+f'_component{k}_vs_'+arm,[o['record']['errors'][axis]['components'][k]/8 for o in options],-np.inf,1.1*control['components'][k]+.01)
                constraint(axis+f'_coverage{k}_vs_'+arm,[o['record']['errors'][axis]['supported_windows'][k] for o in options],control['supported_windows'][k],np.inf)
    objective=np.array([o['record']['errors']['geometry']['mean']/8 for o in options])
    solved=milp(objective,integrality=np.ones(n),bounds=Bounds(0,1),constraints=LinearConstraint(np.array(rows),lower,upper),options={'time_limit':60.,'mip_rel_gap':0.})
    chosen=[];checks={};aggregate={}
    if solved.x is not None:
        assert np.max(np.abs(solved.x-np.round(solved.x)))<1e-6
        selected=[o for o,x in zip(options,solved.x) if x>.5]
        assert len(selected)==8 and sorted(o['family_index'] for o in selected)==list(range(8))
        checks={'complete_panel':True,'six_nonfallback':sum(o['seed'] is not None for o in selected)>=6}
        for axis in ('rhythm','geometry'):
            data=[o['record']['errors'][axis] for o in selected]
            aggregate[axis]={'mean':float(np.mean([d['mean'] for d in data])),
                'components':np.mean([d['components'] for d in data],axis=0).tolist(),
                'supported_windows':np.sum([d['supported_windows'] for d in data],axis=0).tolist()}
            for arm in ('b0','retrieval'):
                a,b=aggregate[axis],report['aggregate'][arm][axis]
                checks[axis+'_10percent_vs_'+arm]=a['mean']<=.9*b['mean'] and a['mean']<b['mean']
                checks[axis+'_components_vs_'+arm]=all(x<=1.1*y+.01 for x,y in zip(a['components'],b['components']))
                checks[axis+'_coverage_vs_'+arm]=all(x>=y for x,y in zip(a['supported_windows'],b['supported_windows']))
        checks['per_song_support_and_contradictions']=all(allowed(o['record'],songs[o['family_index']]['b0'],songs[o['family_index']]['arms']['retrieval']['selected']) for o in selected)
        chosen=[{'family':songs[o['family_index']]['family'],'oracle_seed':o['seed']} for o in selected]
    feasible=bool(checks) and all(checks.values())
    result={'identity':frozen['identity'],'status':'ORACLE_FEASIBLE_NOT_SERVABLE' if feasible else 'ORACLE_INFEASIBLE' if solved.status==2 else 'ORACLE_UNRESOLVED',
        'solver_status':int(solved.status),'eligible_options':n,'gates':checks,'aggregate':aggregate,'reference_opened_witness':chosen,
        'claim':'reference-dependent existence bound only; no learned selector, real selected output, candidate pass or production change'}
    path=OUT/'report.json'
    if path.exists():assert json.loads(path.read_text())==result
    _atomic(path,result);return result


if __name__=='__main__':print(json.dumps(run(),indent=1))
