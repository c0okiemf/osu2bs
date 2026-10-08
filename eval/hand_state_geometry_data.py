"""Authenticated hand-state tensors and fixed native validation templates."""
from collections import Counter
import json
import time

import numpy as np
import torch

from eval import hand_state_geometry as geometry,phrase_native_phase as phase,phrase_native as native
from eval import phrase_budget as budget,joint_diverse_fit as corpus,joint_approved_fit as approved,joint_uniform_audio as uniform,joint_pilot as old
from eval.expressive_manifest import freeze_run
from eval.joint_model import JointModel,_torch_save
from eval.joint_phrase import ROOT,_atomic,_sha,verify_sources
from eval.joint_export import export_chart,read_chart,assert_same

OUT=ROOT/'experiments/hand-state-geometry-v1'
DATA=OUT/'data-stage'
SPEC=ROOT/'docs/specs/2026-09-30-hand-state-geometry-design.md'


def freeze():
    parent=phase.freeze();audit=json.loads((phase.OUT/'artifact-audit.json').read_text())
    if audit['repeat_report_sha256']!=_sha(phase.OUT/'report.json'):raise ValueError('phase result not audited')
    prior=json.loads((corpus.OUT/'run.json').read_text());config=prior['config'];sources=config['sources']
    if Counter(r['role'] for r in sources.values())!={'train':336,'validation':6} or 'fam:24227' in sources:raise ValueError('source split changed')
    if {f for f,r in sources.items() if r['role']=='validation'}!=approved.VALIDATION:raise ValueError('validation membership changed')
    inputs={}
    for family,item in sorted(sources.items()):
        verify_sources(item['record'])
        if _sha(ROOT/item['payload'])!=item['payload_sha256']:raise ValueError('source payload changed')
        root=uniform.OUT/'audio' if family in config['uniform_views'] else approved.OUT/'audio'
        path=root/family.replace(':','_')/'features.pt'
        receipt=config['uniform_views'].get(family,config['validation_views'].get(family))
        if _sha(path)!=receipt['features_sha256']:raise ValueError('source audio view changed')
        inputs[family]={**item,'view':str(path.relative_to(ROOT)),'view_sha256':receipt['features_sha256']}
    calibration=json.loads((approved.OUT/'prepared.json').read_text())
    return freeze_run(DATA,{'parent_identity':parent['identity'],'parent_audit_sha256':_sha(phase.OUT/'artifact-audit.json'),
        'parent_report_sha256':_sha(phase.OUT/'report.json'),'sources':inputs,'corpus_run_sha256':_sha(corpus.OUT/'run.json'),
        'code_sha256':_sha(__file__),'model_code_sha256':_sha(geometry.__file__),'spec_sha256':_sha(SPEC),
        'validation_rate':calibration['validation_rate'],'validation_scales':calibration['validation_scales'],
        'calibration_sha256':_sha(approved.OUT/'prepared.json'),'protected':parent['config']['protected'],
        'control_checkpoint':parent['config']['checkpoint'],'control_checkpoint_sha256':parent['config']['checkpoint_sha256'],
        'budget_model_sha256':_sha(budget.OUT/'model.pt'),'native_bank_sha256':_sha(uniform.OUT/'retrieval.pt'),
        'dependencies':{p:_sha(ROOT/p) for p in ('eval/phrase_native.py','eval/phrase_budget.py','eval/phrase_native_phase.py',
            'eval/joint_pilot.py','eval/joint_model.py','eval/joint_decode.py','eval/joint_export.py','eval/joint_phrase.py')}})


def verify_record(template,actual,record,baseline):
    phase.verify_export(template,actual,record,baseline)


def measure(attempt,template,data,dest,identity,scales):
    attempt['signature_ok']=phase.base.typed.signature(attempt['source'])==phase.base.typed.signature(template)
    attempt['phase_signature_ok']=geometry.signature(attempt['source'])==geometry.signature(template)
    attempt['direction_counters_ok']=geometry.direction_counters(attempt['source'])==geometry.direction_counters(template)
    record=old.measure_attempt(attempt,data,dest,identity,scales)
    if record['ok']:
        verify_record(template,read_chart(dest/'ExpertPlus.dat',dest/'Info.dat'),record,{'profile':old.profile(template,data['audio'])})
    return record


def aggregate(records):
    complete=[r for r in records if r['ok']]
    values=[r['errors']['geometry'] for r in complete]
    valid=bool(values) and all(v['mean'] is not None and all(x is not None for x in v['components']) for v in values)
    return {'attempts':len(records),'failures':len(records)-len(complete),'geometry':{
        'mean':float(np.mean([v['mean'] for v in values])) if valid else None,
        'components':np.mean([v['components'] for v in values],axis=0).tolist() if valid else [None]*5,
        'supported_windows':np.sum([v['supported_windows'] for v in values],axis=0).tolist() if values else [0]*5}}


def load_validation(frozen,family):
    item=frozen['config']['sources'][family]
    if item['role']!='validation':raise ValueError('not validation')
    key=family.replace(':','_');root=DATA/'validation'/key
    saved=json.loads((root/'template.json').read_text())
    actual=read_chart(root/'template'/'ExpertPlus.dat',root/'template'/'Info.dat')
    for name,sha in saved['artifacts'].items():
        if _sha(root/'template'/name)!=sha:raise ValueError('template artifact changed')
    actual['duration_beats']=saved['duration_beats']
    return torch.load(ROOT/item['payload'],weights_only=False),torch.load(ROOT/item['view'],weights_only=False),actual


def prepare():
    torch.set_num_threads(2);deadline=time.monotonic()+2700;frozen=freeze();receipts={};controls=[]
    planner=torch.load(budget.OUT/'model.pt',weights_only=False);bank=torch.load(uniform.OUT/'retrieval.pt',weights_only=False)
    control=JointModel().eval();control.load_state_dict(torch.load(ROOT/frozen['config']['control_checkpoint'],map_location='cpu',weights_only=True))
    for family,item in frozen['config']['sources'].items():
        key=family.replace(':','_')
        if time.monotonic()>=deadline:return {'status':'INCOMPLETE','family':family}
        data=torch.load(ROOT/item['payload'],weights_only=False);audio=torch.load(ROOT/item['view'],weights_only=False)
        if item['role']=='train':
            path=DATA/'train'/(key+'.pt');rp=path.with_suffix('.json')
            if not rp.exists():
                tensors=geometry.teacher(data['source'],audio);_torch_save(path,tensors)
                _atomic(rp,{'identity':frozen['identity'],'sha256':_sha(path),'events':len(tensors['count'])})
                print('state geometry teacher',family,len(tensors['count']),flush=True)
            receipt=json.loads(rp.read_text())
            if receipt['identity']!=frozen['identity'] or receipt['sha256']!=_sha(path):raise ValueError('teacher identity changed')
            receipts[family]=receipt
            continue
        root=DATA/'validation'/key;tp=root/'template.json'
        if not tp.exists():
            source={k:data['source'][k] for k in ('bpm','duration_beats')}
            source.update(notes=[],events=[],bombs=[],walls=[])
            rate=frozen['config']['validation_rate'];plans=native.plan(source,audio,planner,rate);selected=None
            for seed in range(6):
                attempt=native.render(source,audio,bank,plans,rate,seed)
                candidate=attempt.pop('source',None);_atomic(root/'native-attempts'/f'{seed}.json',attempt)
                if attempt['ok']:
                    selected=seed;actual=export_chart(candidate,root/'template');assert_same(candidate,actual,18,0);break
            if selected is None:
                result={'identity':frozen['identity'],'status':'VALIDATION_TEMPLATE_FAILURE','family':family}
                _atomic(DATA/'preparation-failure.json',result);return result
            _atomic(tp,{'identity':frozen['identity'],'selected_seed':selected,'duration_beats':source['duration_beats'],
                'artifacts':{name:_sha(root/'template'/name) for name in ('ExpertPlus.dat','Info.dat')},
                'plans':plans,'selection':'first complete native attempt; no reference geometry or QA'})
        saved=json.loads(tp.read_text())
        if saved['identity']!=frozen['identity']:raise ValueError('template identity changed')
        data,audio,template=load_validation(frozen,family)
        for seed in range(2):
            path=root/'control'/str(seed);record=old.cached_record(path/'record.json',frozen['identity'])
            if record is None:
                attempt=phase.refine(control,template,audio,seed,1.)
                record=measure(attempt,template,data,path,frozen['identity'],frozen['config']['validation_scales'])
                print('state geometry validation control',family,seed,record['ok'],flush=True)
            controls.append(record)
        receipts[family]={'template_sha256':_sha(tp),'artifacts':saved['artifacts'],
            'control_sha256':[_sha(root/'control'/str(seed)/'record.json') for seed in range(2)]}
    result={'identity':frozen['identity'],'status':'PREPARED','receipts':receipts,'control':aggregate(controls),
        'training_events':sum(r['events'] for r in receipts.values() if 'events' in r)}
    path=DATA/'prepared.json'
    if path.exists():assert json.loads(path.read_text())==result
    _atomic(path,result);freeze();return result


if __name__=='__main__':print(json.dumps(prepare(),indent=1))
