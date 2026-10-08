"""One source-diversity ablation with the same joint model and repaired decoder."""
from collections import Counter
import json
import random
import time
from unittest.mock import patch

import numpy as np
import torch

from eval import joint_approved_fit as fit,joint_aligned_selection as aligned,joint_censored_timing as decoder
from eval import joint_pilot as old,joint_start as start,joint_ordered as ordered,joint_uniform_audio as uniform,joint_recovered_inventory as recovered
from eval.expressive_manifest import freeze_run
from eval.joint_model import KEYS,SEED,SNAPSHOTS,JointModel,_torch_save
from eval.joint_phrase import ROOT,_atomic,_sha,verify_sources
from eval.joint_export import read_chart
from eval.joint_continuity import describe

OUT=ROOT/'experiments/joint-phrase-v19/diverse-joint'
SPEC=ROOT/'docs/specs/2026-09-30-diverse-joint-fit-design.md'
validation_summary=fit.validation_summary
choose_snapshot=fit.choose_snapshot
gap_diagnostic=fit.gap_diagnostic


def inventory():
    original=json.loads((old.OUT/'run.json').read_text())['config']['records']
    revised=json.loads((recovered.OUT/'inventory.json').read_text())
    additions=json.loads((recovered.OUT/'run.json').read_text())['config']['new_sources']
    result={}
    for record,root in [(r,old.OUT) for r in original]+[(r['composite_record'],recovered.OUT) for r in additions]:
        family=record['family'];role=revised['roles']['sources'][family]['future_role']
        if role!='train' and not (role=='validation' and family in fit.VALIDATION):continue
        verify_sources(record);path=root/'data'/(family.replace(':','_')+'.pt');receipt=json.loads(path.with_suffix('.json').read_text())
        if _sha(path)!=receipt['sha256']:raise ValueError('source payload changed')
        result[family]={'role':role,'record':record,'payload':str(path.relative_to(ROOT)),'payload_sha256':receipt['sha256'],
                        'receipt_sha256':_sha(path.with_suffix('.json'))}
    if Counter(r['role'] for r in result.values())!={'train':336,'validation':6}:raise ValueError('diverse inventory changed')
    if {f for f,r in result.items() if r['role']=='validation'}!=fit.VALIDATION or 'fam:24227' in result:raise ValueError('held-out role changed')
    expected=json.loads((uniform.OUT/'run.json').read_text())['config']['source_inventory']
    if {f for f,r in result.items() if r['role']=='train'}!={f for f,r in expected.items() if r['role']=='train'}:raise ValueError('uniform train membership differs')
    return result


def freeze_experiment():
    parent,approved_run=aligned.freeze_experiment();report=json.loads((aligned.OUT/'report.json').read_text())
    if report['identity']!=parent['identity'] or report['status'] not in ('DEVELOPMENT_NEGATIVE','NO_LEARNED_CANDIDATE'):raise ValueError('start condition: aligned approved-only negative required')
    audit=json.loads((aligned.OUT/'artifact-audit.json').read_text())
    if audit['repeat_report_sha256']!=_sha(aligned.OUT/'report.json'):raise ValueError('aligned control audit changed')
    approved_run,prepared=fit.prepare();sources=inventory();views=json.loads((uniform.OUT/'bank-receipt.json').read_text())['audio_receipts']
    for family,r in views.items():
        root=uniform.OUT/'audio'/family.replace(':','_')
        if _sha(root/'features.pt')!=r['features_sha256'] or _sha(root/'common14800.wav')!=r['render']['sha256']:raise ValueError('uniform source view changed')
    approved_tensors={f:r['teacher'] for f,r in prepared['receipts'].items() if 'teacher' in r}
    if len(approved_tensors)!=45 or any(sources[f]['role']!='train' for f in approved_tensors):raise ValueError('approved subset changed')
    return freeze_run(OUT,{'parent_identity':parent['identity'],'approved_model_identity':approved_run['identity'],'sources':sources,'uniform_views':views,
        'approved_tensors':approved_tensors,'validation_views':{f:r['view'] for f,r in prepared['receipts'].items() if 'view' in r},
        'validation_scales':prepared['validation_scales'],'validation_rate':prepared['validation_rate'],
        'code_sha256':_sha(__file__),'spec_sha256':_sha(SPEC),'decoder_sha256':_sha(decoder.__file__),
        'approved_prepared_sha256':_sha(fit.OUT/'prepared.json'),'control_report_sha256':_sha(aligned.OUT/'report.json'),
        'control_audit_sha256':_sha(aligned.OUT/'artifact-audit.json'),'training':approved_run['config']['training'],
        'control_artifacts':{str(p.relative_to(ROOT)):_sha(p) for p in (aligned.OUT/'development').rglob('*')
            if p.is_file() and p.name in ('record.json','song.json','ExpertPlus.dat','Info.dat')},
        'selection':approved_run['config']['selection']})


def load_view(family,frozen):
    root=uniform.OUT/'audio'/family.replace(':','_') if family in frozen['config']['uniform_views'] else fit.OUT/'audio'/family.replace(':','_')
    return torch.load(root/'features.pt',weights_only=False)


def teacher_path(family,frozen):
    root=fit.OUT if family in frozen['config']['approved_tensors'] else OUT
    return root/'data'/(family.replace(':','_')+'.pt')


def prepare(deadline=None):
    frozen=freeze_experiment();receipts={}
    for family,item in sorted(frozen['config']['sources'].items()):
        if item['role']!='train':continue
        path=teacher_path(family,frozen);rp=path.with_suffix('.json')
        if family in frozen['config']['approved_tensors']:
            r=frozen['config']['approved_tensors'][family]
            if _sha(path)!=r['sha256']:raise ValueError('original approved tensor changed')
        else:
            if not rp.exists():
                if deadline is not None and time.monotonic()>=deadline:raise TimeoutError('diverse teacher preparation deadline')
                data=torch.load(ROOT/item['payload'],weights_only=False);tensors=fit.teacher(data,load_view(family,frozen),item['role'])
                if len(tensors['gap'])<128:raise ValueError('train source too short')
                _torch_save(path,tensors);_atomic(rp,{'identity':frozen['identity'],'sha256':_sha(path),'actions':len(tensors['gap'])})
                print('diverse teacher prepared',family,len(tensors['gap']),flush=True)
            r=json.loads(rp.read_text())
            if r['identity']!=frozen['identity'] or _sha(path)!=r['sha256']:raise ValueError('diverse teacher cache changed')
        receipts[family]=r
    result={'identity':frozen['identity'],'receipts':receipts,'validation_scales':frozen['config']['validation_scales'],'validation_rate':frozen['config']['validation_rate']}
    rp=OUT/'prepared.json'
    if rp.exists() and json.loads(rp.read_text())!=result:raise ValueError('prepared corpus changed')
    _atomic(rp,result);return frozen,result


def train(deadline_seconds=2700):
    deadline=time.monotonic()+min(deadline_seconds,2700);torch.set_num_threads(4)
    try:frozen,prepared=prepare(deadline)
    except TimeoutError:return {'status':'INCOMPLETE','stage':'prepare'}
    rows=[torch.load(teacher_path(f,frozen),weights_only=False) for f,r in sorted(frozen['config']['sources'].items()) if r['role']=='train']
    if len(rows)!=336:raise ValueError('train membership changed')
    random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');model=JointModel().to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.01);cursor,bos_crops,history=0,0,[];path=OUT/'train_state.pt'
    if path.exists():
        saved=torch.load(path,map_location=device,weights_only=False)
        if saved['identity']!=frozen['identity']:raise ValueError('training identity changed')
        model.load_state_dict(saved['model']);opt.load_state_dict(saved['optimizer']);random.setstate(saved['python_rng'])
        np.random.set_state(saved['numpy_rng']);torch.set_rng_state(saved['torch_rng'].cpu())
        if torch.cuda.is_available():torch.cuda.set_rng_state_all([x.cpu() for x in saved['cuda_rng']])
        cursor,bos_crops,history=saved['update'],saved['bos_crops'],saved['history']
    if cursor==0 and not (OUT/'snapshot-0.pt').exists():_torch_save(OUT/'snapshot-0.pt',{k:v.cpu() for k,v in model.state_dict().items()})
    model.train()
    while cursor<6000 and time.monotonic()<deadline:
        crops=[]
        for i in range(16):
            row=random.choice(rows);offset=0 if i<4 else random.randrange(len(row['gap'])-128+1)
            bos_crops+=int(offset==0);crops.append(start.crop(row,offset))
        batch={k:torch.stack([c[k] for c in crops]).to(device) for k in (*KEYS,'valid')}
        opt.zero_grad();out,_=model(batch);loss=start.losses(out,batch)
        if not torch.isfinite(loss):raise ValueError('nonfinite training loss')
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step();cursor+=1
        if cursor%100==0:
            history.append({'update':cursor,'loss':float(loss),'bos_crops':bos_crops})
            print(f'update {cursor}/6000 loss={float(loss):.4f} BOS={bos_crops}',flush=True)
        if cursor in SNAPSHOTS:_torch_save(OUT/f'snapshot-{cursor}.pt',{k:v.cpu() for k,v in model.state_dict().items()})
        if cursor%100==0 or cursor==6000 or time.monotonic()>=deadline:
            _torch_save(path,{'identity':frozen['identity'],'update':cursor,'bos_crops':bos_crops,'history':history,
                'model':model.state_dict(),'optimizer':opt.state_dict(),'python_rng':random.getstate(),'numpy_rng':np.random.get_state(),
                'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []})
            _atomic(OUT/'history.json',{'update':cursor,'bos_crops':bos_crops,'history':history})
    return {'status':'FIT_COMPLETE' if cursor==6000 else 'INCOMPLETE','update':cursor,'bos_crops':bos_crops}


def validate(deadline_seconds=2700):
    deadline=time.monotonic()+min(deadline_seconds,2700);torch.set_num_threads(2);frozen,prepared=prepare(deadline);summaries=[]
    for update in SNAPSHOTS:
        path=OUT/f'snapshot-{update}.pt'
        if not path.exists():return {'status':'WAITING_FOR_SNAPSHOT','update':update}
        model=JointModel().eval();model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True));records=[]
        identity=frozen['identity']+':'+_sha(path)
        for family,item in sorted(frozen['config']['sources'].items()):
            if item['role']!='validation':continue
            data=torch.load(ROOT/item['payload'],weights_only=False);audio=load_view(family,frozen)
            for seed in range(2):
                target=OUT/'validation'/str(update)/family.replace(':','_')/str(seed)
                rec=old.cached_record(target/'record.json',identity)
                if rec is None:
                    if time.monotonic()>=deadline:return {'status':'INCOMPLETE','stage':'validation','update':update,'family':family,'seed':seed}
                    attempt=decoder.rollout(model,old.generation_source(data),audio,prepared['validation_rate'],seed=seed)
                    rec=old.measure_attempt(attempt,data,target,identity,prepared['validation_scales'])
                    print('validation',update,family,seed,rec['ok'],rec['reason'],flush=True)
                records.append(rec)
        summary=validation_summary(update,_sha(path),records);_atomic(OUT/'validation'/f'summary-{update}.json',summary);summaries.append(summary)
    chosen=choose_snapshot(summaries)
    result={'identity':frozen['identity'],'status':'VALIDATION_COMPLETE','snapshots':summaries,'selected':chosen['update'],
            'snapshot_sha256':chosen['snapshot_sha256'],'learned_candidate':chosen['update']>0,'selection':frozen['config']['selection']}
    _atomic(OUT/'selection.json',result);return result


def development(deadline_seconds=2700):
    deadline=time.monotonic()+min(deadline_seconds,2700);torch.set_num_threads(2);frozen,prepared=prepare(deadline)
    selection=json.loads((OUT/'selection.json').read_text());path=OUT/f"snapshot-{selection['selected']}.pt"
    if selection['identity']!=frozen['identity'] or _sha(path)!=selection['snapshot_sha256']:raise ValueError('selected checkpoint changed')
    if selection['selected']==0:
        result={'identity':frozen['identity'],'status':'NO_LEARNED_CANDIDATE','release_eligible':False,'selection':selection}
        _atomic(OUT/'report.json',result);return result
    model=JointModel().eval();model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True))
    identity=frozen['identity']+':'+selection['snapshot_sha256'];kit=old.machine_tools();scales=old.evaluation_freeze()['config']['scales'];songs=[];evidence={}
    for family,item in ordered.inputs().items():
        key=family.replace(':','_');root=OUT/'development'/key;records=[]
        prior=json.loads((uniform.OUT/'development'/key/'song.json').read_text());audio=load_view(family,frozen)
        for seed,temp in enumerate((.85,1.,1.15,.85,1.,1.15)):
            target=root/'ordered'/str(seed);rec=old.cached_record(target/'record.json',identity)
            if rec is None:
                if time.monotonic()>=deadline:return {'status':'INCOMPLETE','stage':'development','family':family,'seed':seed}
                attempt=decoder.rollout(model,item['source'],audio,item['rate'],seed=seed,temperature=temp)
                rec=ordered.measure(attempt,item['data'],target,identity,scales,kit)
                print('development',family,seed,rec['ok'],rec.get('machine',{}).get('verdict',rec['reason']),flush=True)
            records.append(rec)
        chosen=ordered.select(records,item['b0']);song={**prior,'identity':identity,'arms':{'retrieval':prior['arms']['retrieval'],'ordered':chosen}}
        _atomic(root/'song.json',song);songs.append(song)
        cp=old.PILOT/'development'/key/'b0' if chosen['fallback'] else root/'ordered'/str(chosen['selected_seed'])
        actual=read_chart(cp/'ExpertPlus.dat',cp/'Info.dat')
        evidence[family]={'selected':describe(actual),'selected_gaps':gap_diagnostic(actual),
            'attempt_gaps':[gap_diagnostic(read_chart(root/'ordered'/str(i)/'ExpertPlus.dat',root/'ordered'/str(i)/'Info.dat')) if r['ok'] else None for i,r in enumerate(records)]}
    freeze_experiment();return decide(songs,frozen,selection,evidence)

def decide(songs,frozen,selection,evidence):
    with patch.object(fit,'OUT',OUT):report=fit.decide(songs,frozen,selection,evidence)
    control_root=aligned.OUT
    prior=json.loads((aligned.OUT/'report.json').read_text())
    if 'inherited_result' in prior:control_root=ROOT/prior['inherited_result']['root']
    differences={}
    for axis in ('rhythm','geometry'):
        paired={}
        for song in songs:
            path=control_root/'development'/song['family'].replace(':','_')/'song.json'
            if not path.exists():paired[song['family']]=None;continue
            control=json.loads(path.read_text());a=song['arms']['ordered']['selected']['errors'][axis]['mean'];b=control['arms']['ordered']['selected']['errors'][axis]['mean']
            paired[song['family']]=a-b if a is not None and b is not None else None
        differences[axis]={'paired_differences':paired,'mean_difference':float(np.mean(list(paired.values()))) if all(x is not None for x in paired.values()) else None}
    report.update(source_diversity_ablation=differences,training_sources={'approved':45,'general_unlabeled':291,'total':336},
        budget='one6000-update fit;48 validation and at most48 development attempts; fixed decoder/calibration/selection rule',
        historical_exposure='new weights fit336 eligible train families only; B0, original retrieval, QA and historical development scales retain prior exposure',
        claim='source-coverage ablation under repaired pipeline; opened development, not confirmation or a quality guarantee')
    report['arm_semantics']['ordered']='fresh diverse336 joint model; same uniform audio, censored decoder and approved6 selection'
    _atomic(OUT/'report.json',report);return report


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=['prepare','train','validate','development'])
    parser.add_argument('--deadline-seconds',type=float,default=2700);args=parser.parse_args()
    result=prepare()[1] if args.command=='prepare' else {'train':train,'validate':validate,'development':development}[args.command](args.deadline_seconds)
    print(json.dumps(result,indent=1))
