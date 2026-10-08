"""One fresh approved-corpus continuous joint fit, with isolated validation selection."""
from collections import Counter
import json
from pathlib import Path
import random
import time
from unittest.mock import patch

import numpy as np
import torch

from eval import joint_pilot as old,joint_start as start,joint_timing as timing,joint_ordered as ordered
from eval import joint_uniform_audio as uniform,joint_audio_view as view,joint_recovered_inventory as recovered,joint_approved as approved
from eval.expressive_manifest import freeze_run
from eval.joint_model import KEYS,SEED,SNAPSHOTS,JointModel,actions,action_notes,context,source_rate,audio_features,_torch_save
from eval.joint_phrase import ROOT,PROTECTED,_atomic,_sha,verify_sources,encode_events
from eval.joint_export import read_chart
from eval.joint_continuity import describe

OUT=ROOT/'experiments/joint-phrase-v16/approved-joint'
SPEC=ROOT/'docs/specs/2026-09-30-approved-joint-fit-design.md'
VALIDATION={'fam:30fd','fam:12db3','fam:1a4dd','fam:21f92','fam:a253','fam:b7aa'}


def inventory():
    original=json.loads((old.OUT/'run.json').read_text())['config']['records']
    revised=json.loads((recovered.OUT/'inventory.json').read_text())
    if revised['report_sha256']!=_sha(recovered.OUT/'report.json'):raise ValueError('source role report changed')
    additions=json.loads((recovered.OUT/'run.json').read_text())['config']['new_sources']
    candidates=[(r,r,old.OUT) for r in original if r['approved']]
    candidates += [(r['composite_record'],r['approved_record'],recovered.OUT) for r in additions]
    result={}
    for record,provenance,root in candidates:
        family=record['family'];role=revised['roles']['sources'][family]['future_role']
        if role not in ('train','validation'):continue
        approved.approved_records([{**provenance,'fit_role':'train'}]);verify_sources(record);verify_sources(provenance)
        path=root/'data'/(family.replace(':','_')+'.pt');receipt=json.loads(path.with_suffix('.json').read_text())
        if _sha(path)!=receipt['sha256']:raise ValueError('source payload changed')
        result[family]={'role':role,'record':record,'provenance':provenance,'payload':str(path.relative_to(ROOT)),
                        'payload_sha256':receipt['sha256'],'receipt_sha256':_sha(path.with_suffix('.json'))}
    if Counter(r['role'] for r in result.values())!={'train':45,'validation':6}:raise ValueError('approved fit membership changed')
    if {f for f,r in result.items() if r['role']=='validation'}!=VALIDATION or 'fam:24227' in result:raise ValueError('reserved roles changed')
    bank=torch.load(uniform.OUT/'retrieval.pt',weights_only=False)
    if {e['family'] for e in bank['entries']}!={f for f,r in result.items() if r['role']=='train'}:raise ValueError('train source set differs from approved45')
    return result


def freeze_experiment():
    old.evaluation_freeze()
    protected={p:_sha(ROOT/p) for p in PROTECTED}
    if protected!=json.loads((old.OUT/'run.json').read_text())['config']['protected']:raise ValueError('B0 changed')
    report=json.loads((uniform.OUT/'report.json').read_text());receipt=json.loads((uniform.OUT/'bank-receipt.json').read_text())
    if not report['gates']['complete_panel'] or receipt['identity']!=report['identity']:raise ValueError('uniform control incomplete')
    sources=inventory();views={}
    for family,r in receipt['audio_receipts'].items():
        if family not in sources and family not in {x['family'] for x in report['families']}:continue
        root=uniform.OUT/'audio'/family.replace(':','_')
        if _sha(root/'features.pt')!=r['features_sha256'] or _sha(root/'common14800.wav')!=r['render']['sha256']:raise ValueError('uniform view changed')
        views[family]=r
    return freeze_run(OUT,{'code_sha256':_sha(__file__),'spec_sha256':_sha(SPEC),'sources':sources,'uniform_views':views,'protected':protected,
        'inventory_sha256':_sha(recovered.OUT/'inventory.json'),'recovery_proofs_sha256':_sha(recovered.OUT/'run.json'),
        'uniform_report_sha256':_sha(uniform.OUT/'report.json'),'uniform_bank_receipt_sha256':_sha(uniform.OUT/'bank-receipt.json'),
        'control_artifacts':{str(p.relative_to(ROOT)):_sha(p) for p in (uniform.OUT/'development').rglob('*')
            if p.is_file() and p.name in ('song.json','record.json','ExpertPlus.dat','Info.dat')},
        'dependencies':{p:_sha(ROOT/p) for p in ('eval/joint_model.py','eval/joint_start.py','eval/joint_timing.py',
            'eval/joint_decode.py','eval/joint_phrase.py','eval/joint_export.py','eval/joint_pilot.py','eval/joint_ordered.py',
            'eval/joint_deployment.py','eval/joint_audio_view.py','eval/joint_continuity.py','parity.py','motion.py','eval/quality_metrics.py')},
        'training':{'seed':SEED,'updates':6000,'snapshots':SNAPSHOTS,'batch':16,'bos':4,'sequence':128,'warmup':32,'lr':3e-4,'weight_decay':.01},
        'selection':'failures; mean rhythm+geometry; earliest update'})


def teacher(data,audio,role):
    if role!='train':raise ValueError('only train sources may produce fit tensors')
    source=data['source'];rows=actions(source)
    if action_notes(rows)!=list(map(tuple,source['notes'])):raise ValueError('teacher action roundtrip changed')
    rate=source_rate(source)
    ctx=np.stack([context(audio,r['cursor'],source['bpm'],rate,source['authored']['njs'],source['walls'],source['bombs']) for r in rows])
    return {'context':torch.from_numpy(ctx),'prev':torch.tensor([r['prev_slots'] for r in rows]),
        'kind':torch.tensor([r['prev_kind'] for r in rows]),'prev_gap':torch.tensor([r['prev_gap'] for r in rows],dtype=torch.float32),
        'gap':torch.tensor([r['gap'] for r in rows]),'residual':torch.tensor([r['residual'] for r in rows],dtype=torch.float32),
        'count':torch.tensor([r['count'] for r in rows]),'slots':torch.tensor([r['slots'] for r in rows])}


def load_view(family,frozen):
    root=uniform.OUT/'audio'/family.replace(':','_') if family in frozen['config']['uniform_views'] else OUT/'audio'/family.replace(':','_')
    return torch.load(root/'features.pt',weights_only=False)


def prepare(deadline=None):
    frozen=freeze_experiment();receipts={};profiles=[];rates=[]
    for family,item in sorted(frozen['config']['sources'].items()):
        key=family.replace(':','_');root=OUT/'audio'/key
        if family not in frozen['config']['uniform_views']:
            rp=root/'receipt.json'
            if not rp.exists():
                if deadline is not None and time.monotonic()>=deadline:raise TimeoutError('validation audio deadline')
                render=view.render_common(item['record']['sources']['audio']['path'],root/'common14800.wav')
                _torch_save(root/'features.pt',audio_features(root/'common14800.wav'))
                _atomic(rp,{'identity':frozen['identity'],'render':render,'features_sha256':_sha(root/'features.pt')})
            r=json.loads(rp.read_text())
            if r['identity']!=frozen['identity'] or r['features_sha256']!=_sha(root/'features.pt') or r['render']['sha256']!=_sha(root/'common14800.wav'):raise ValueError('validation view changed')
            receipts[family]={'view':r}
        if item['role']!='train':continue
        data=torch.load(ROOT/item['payload'],weights_only=False);path=OUT/'data'/(key+'.pt');rp=path.with_suffix('.json')
        if not rp.exists():
            if deadline is not None and time.monotonic()>=deadline:raise TimeoutError('teacher preparation deadline')
            tensors=teacher(data,load_view(family,frozen),item['role'])
            if len(tensors['gap'])<128:raise ValueError('training source too short')
            _torch_save(path,tensors);_atomic(rp,{'identity':frozen['identity'],'sha256':_sha(path),'actions':len(tensors['gap'])})
            print('approved teacher prepared',family,len(tensors['gap']),flush=True)
        r=json.loads(rp.read_text())
        if r['identity']!=frozen['identity'] or _sha(path)!=r['sha256']:raise ValueError('teacher cache changed')
        receipts[family]={'teacher':r};profiles.append(old.profile(data['source'],data['audio']));rates.append(source_rate(data['source']))
    result={'identity':frozen['identity'],'receipts':receipts,'validation_scales':old.fit_scales(profiles),'validation_rate':float(np.median(rates))}
    rp=OUT/'prepared.json'
    if rp.exists() and json.loads(rp.read_text())!=result:raise ValueError('prepared selection inputs changed')
    _atomic(rp,result);return frozen,result


def train(deadline_seconds=2700):
    deadline=time.monotonic()+min(deadline_seconds,2700);torch.set_num_threads(4)
    try:frozen,prepared=prepare(deadline)
    except TimeoutError:return {'status':'INCOMPLETE','stage':'prepare'}
    rows=[torch.load(OUT/'data'/(f.replace(':','_')+'.pt'),weights_only=False) for f,r in sorted(frozen['config']['sources'].items()) if r['role']=='train']
    if len(rows)!=45:raise ValueError('train membership changed')
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


def validation_summary(update,snapshot_sha,records):
    good=[r for r in records if r['ok']]
    values=[sum(r['errors'][a]['mean'] for a in ('rhythm','geometry')) for r in good if all(r['errors'][a]['mean'] is not None for a in ('rhythm','geometry'))]
    return {'update':update,'snapshot_sha256':snapshot_sha,'attempts':len(records),'complete':len(good),'failures':len(records)-len(good),
            'failure_reasons':dict(Counter(r['reason'] for r in records if not r['ok'])),
            'error':float(np.mean(values)) if good and len(values)==len(good) else None}


def choose_snapshot(summaries):
    return min(summaries,key=lambda r:(r['failures'],float('inf') if r['error'] is None else r['error'],r['update']))


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
                    attempt=timing.rollout(model,old.generation_source(data),audio,prepared['validation_rate'],seed=seed)
                    rec=old.measure_attempt(attempt,data,target,identity,prepared['validation_scales'])
                    print('validation',update,family,seed,rec['ok'],rec['reason'],flush=True)
                records.append(rec)
        summary=validation_summary(update,_sha(path),records);_atomic(OUT/'validation'/f'summary-{update}.json',summary);summaries.append(summary)
    chosen=choose_snapshot(summaries)
    result={'identity':frozen['identity'],'status':'VALIDATION_COMPLETE','snapshots':summaries,'selected':chosen['update'],
            'snapshot_sha256':chosen['snapshot_sha256'],'learned_candidate':chosen['update']>0,'selection':frozen['config']['selection']}
    _atomic(OUT/'selection.json',result);return result


def gap_diagnostic(source):
    gaps=[]
    for hand in (0,1):
        beats=sorted({n[0] for n in source['notes'] if n[1]==hand});gaps.extend((b-a)*60000/source['bpm'] for a,b in zip(beats,beats[1:]))
    return {'same_hand_intervals':len(gaps),'minimum_ms':min(gaps,default=None),'under_1ms':sum(x<1 for x in gaps),'under_10ms':sum(x<10 for x in gaps)}


def decide(songs,frozen,selection,evidence):
    with patch.object(ordered,'OUT',OUT):report=ordered.decide(songs,frozen)
    controls={s['family']:json.loads((uniform.OUT/'development'/s['family'].replace(':','_')/'song.json').read_text()) for s in songs}
    report['continuous_vs_uniform_retrieval']={}
    for axis in ('rhythm','geometry'):
        delta={s['family']:s['arms']['ordered']['selected']['errors'][axis]['mean']-controls[s['family']]['arms']['ordered']['selected']['errors'][axis]['mean']
               if s['arms']['ordered']['selected']['errors'][axis]['mean'] is not None else None for s in songs}
        report['continuous_vs_uniform_retrieval'][axis]={'paired_differences':delta,'mean_difference':float(np.mean(list(delta.values()))) if all(v is not None for v in delta.values()) else None}
    report.update(selection=selection,continuity_and_gaps=evidence,arm_semantics={'ordered':'fresh approved45 continuous joint model; uniform audio; first admitted',
        'retrieval':'frozen original deployment retrieval','b0':'production baseline'},
        historical_exposure='new weights fit approved45 only; historical B0, retrieval, QA and development scales retain old exposure',
        budget='one6000-update fit;48 validation and48 development attempts; no checkpoint or temperature choice using development',
        claim='opened development pilot; corpus and frontend changed together; not a quality guarantee or confirmation')
    _atomic(OUT/'report.json',report);return report


def development(deadline_seconds=2700):
    deadline=time.monotonic()+min(deadline_seconds,2700);torch.set_num_threads(2);frozen,prepared=prepare(deadline)
    selection=json.loads((OUT/'selection.json').read_text());path=OUT/f"snapshot-{selection['selected']}.pt"
    if selection['identity']!=frozen['identity'] or _sha(path)!=selection['snapshot_sha256']:raise ValueError('selected checkpoint changed')
    model=JointModel().eval();model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True))
    identity=frozen['identity']+':'+selection['snapshot_sha256'];kit=old.machine_tools();scales=old.evaluation_freeze()['config']['scales'];songs=[];evidence={}
    for family,item in ordered.inputs().items():
        key=family.replace(':','_');root=OUT/'development'/key;records=[]
        prior=json.loads((uniform.OUT/'development'/key/'song.json').read_text());audio=load_view(family,frozen)
        for seed,temp in enumerate((.85,1.,1.15,.85,1.,1.15)):
            target=root/'ordered'/str(seed);rec=old.cached_record(target/'record.json',identity)
            if rec is None:
                if time.monotonic()>=deadline:return {'status':'INCOMPLETE','stage':'development','family':family,'seed':seed}
                attempt=timing.rollout(model,item['source'],audio,item['rate'],seed=seed,temperature=temp)
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


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=['prepare','train','validate','development'])
    parser.add_argument('--deadline-seconds',type=float,default=2700);args=parser.parse_args()
    result=prepare()[1] if args.command=='prepare' else {'train':train,'validate':validate,'development':development}[args.command](args.deadline_seconds)
    print(json.dumps(result,indent=1))
