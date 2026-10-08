"""One conditional geometry fit; checkpoint selection uses exported rollouts."""
import json
import os
import sys
import time

import numpy as np
import torch

from eval import hand_state_geometry as geometry,hand_state_geometry_data as data,joint_pilot as old
from eval.expressive_manifest import freeze_run
from eval.joint_model import SEED,SNAPSHOTS,_torch_save
from eval.joint_phrase import ROOT,_sha,_atomic

OUT=data.OUT


def freeze():
    parent=data.freeze();prepared=json.loads((data.DATA/'prepared.json').read_text())
    if prepared['identity']!=parent['identity'] or prepared['status']!='PREPARED':raise ValueError('data not prepared')
    control=prepared['control']
    if control['attempts']!=12 or control['failures'] or control['geometry']['mean'] is None:
        raise ValueError('fixed validation control incomplete or unmeasurable; do not fit')
    for family,item in parent['config']['sources'].items():
        r=prepared['receipts'][family];key=family.replace(':','_')
        if item['role']=='train':
            if _sha(data.DATA/'train'/(key+'.pt'))!=r['sha256']:raise ValueError('training tensor changed')
        else:
            root=data.DATA/'validation'/key
            if _sha(root/'template.json')!=r['template_sha256']:raise ValueError('validation template changed')
            for seed,sha in enumerate(r['control_sha256']):
                if _sha(root/'control'/str(seed)/'record.json')!=sha:raise ValueError('control changed')
    frozen=freeze_run(OUT,{'data_identity':parent['identity'],'prepared_sha256':_sha(data.DATA/'prepared.json'),
        'code_sha256':_sha(__file__),'model_code_sha256':_sha(geometry.__file__),'spec_sha256':_sha(data.SPEC),
        'training':{'seed':SEED,'updates':6000,'snapshots':SNAPSHOTS,'batch':512,'lr':3e-4,'weight_decay':.01,'clip':1.,
            'sampling':'uniform family then uniform event','loss':'mean legal-slot CE per event, then batch mean'},
        'selection':'incomplete outputs; mean geometry error; earliest update','validation_seeds':[0,1],'validation_temperature':1.,
        'protected':parent['config']['protected'],'torch_version':torch.__version__,'threads':2,
        'device':'cuda' if torch.cuda.is_available() else 'cpu','dependencies':{'data_code_sha256':_sha(data.__file__)}})
    return frozen,parent,prepared


def train():
    torch.set_num_threads(2);deadline=time.monotonic()+2700;frozen,parent,prepared=freeze()
    families=[f for f,r in parent['config']['sources'].items() if r['role']=='train']
    rows=[torch.load(data.DATA/'train'/(f.replace(':','_')+'.pt'),weights_only=True) for f in families]
    sizes=np.array([len(r['count']) for r in rows]);starts=np.r_[0,sizes.cumsum()[:-1]]
    tensors={k:torch.cat([r[k] for r in rows]) for k in geometry.KEYS};del rows
    if os.sysconf('SC_AVPHYS_PAGES')*os.sysconf('SC_PAGE_SIZE')<2*1024**3:raise MemoryError('less than2GB system headroom')
    torch.manual_seed(SEED);rng=np.random.default_rng(SEED);device=torch.device(frozen['config']['device'])
    model=geometry.GeometryModel().to(device);optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.01)
    cursor=0;history=[];snapshots={};state_path=OUT/'train_state.pt'
    if state_path.exists():
        saved=torch.load(state_path,map_location=device,weights_only=False)
        if saved['identity']!=frozen['identity']:raise ValueError('training resume identity changed')
        model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer'])
        rng.bit_generator.state=saved['numpy_rng'];torch.set_rng_state(saved['torch_rng'].cpu())
        if device.type=='cuda':torch.cuda.set_rng_state_all([v.cpu() for v in saved['cuda_rng']])
        cursor,history,snapshots=saved['update'],saved['history'],saved['snapshots']
        for step,sha in snapshots.items():
            if _sha(OUT/f'snapshot-{step}.pt')!=sha:raise ValueError('training snapshot changed')
    def snapshot(step):
        path=OUT/f'snapshot-{step}.pt';weights={k:v.detach().cpu() for k,v in model.state_dict().items()}
        if path.exists():
            prior=torch.load(path,map_location='cpu',weights_only=True)
            assert prior.keys()==weights.keys() and all(torch.equal(prior[k],v) for k,v in weights.items())
        else:_torch_save(path,weights)
        snapshots[str(step)]=_sha(path)
    def save():
        _torch_save(state_path,{'identity':frozen['identity'],'update':cursor,'model':model.state_dict(),'optimizer':optimizer.state_dict(),
            'numpy_rng':rng.bit_generator.state,'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all() if device.type=='cuda' else [],
            'history':history,'snapshots':snapshots})
    if cursor==0:snapshot(0)
    model.train()
    while cursor<6000 and time.monotonic()<deadline:
        fs=rng.integers(0,len(families),512);indices=torch.from_numpy(starts[fs]+rng.integers(sizes[fs]))
        batch={k:v[indices].to(device) for k,v in tensors.items()}
        optimizer.zero_grad(set_to_none=True);value=geometry.loss(model(batch),batch)
        if not torch.isfinite(value):raise ValueError('nonfinite geometry training loss')
        value.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();cursor+=1
        if cursor%100==0:
            history.append({'update':cursor,'teacher_legal_ce':float(value.detach().cpu())});print('hand-state geometry',history[-1],flush=True)
        if cursor in SNAPSHOTS:snapshot(cursor)
        if cursor%100==0:save()
    save();result={'identity':frozen['identity'],'status':'FIT_COMPLETE' if cursor==6000 else 'INCOMPLETE',
        'updates':cursor,'snapshots':snapshots,'training_events':int(sizes.sum()),'families':len(families),'history':history}
    _atomic(OUT/'training.json',result);return result


def choose(summaries):
    return min(summaries,key=lambda s:(s['failures'],s['geometry']['mean'] if s['geometry']['mean'] is not None else float('inf'),s['update']))


def qualify(selected,control,records):
    a,b=selected['geometry'],control['geometry']
    return {'nonzero_update':selected['update']>0,'complete_validation':selected['attempts']==12 and selected['failures']==0,
        'geometry_10percent_vs_control':a['mean'] is not None and b['mean'] is not None and a['mean']<=.9*b['mean'] and a['mean']<b['mean'],
        'geometry_components_vs_control':all(x is not None and y is not None and x<=1.1*y+.01 for x,y in zip(a['components'],b['components'])),
        'geometry_coverage_vs_control':all(x>=y for x,y in zip(a['supported_windows'],b['supported_windows'])),
        'exact_native_plans':all(r['signature_ok'] and r['phase_signature_ok'] and r['direction_counters_ok'] for r in records)}


def validate():
    torch.set_num_threads(2);deadline=time.monotonic()+2700;frozen,parent,prepared=freeze()
    trained=json.loads((OUT/'training.json').read_text())
    if trained['identity']!=frozen['identity'] or trained['status']!='FIT_COMPLETE':raise ValueError('fit not complete')
    summaries=[];by_update={}
    for update in SNAPSHOTS:
        path=OUT/f'snapshot-{update}.pt';sha=_sha(path)
        if sha!=trained['snapshots'][str(update)]:raise ValueError('snapshot changed')
        identity=frozen['identity']+':'+sha;model=geometry.GeometryModel().eval()
        model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True));records=[]
        for family,item in parent['config']['sources'].items():
            if item['role']!='validation':continue
            reference,audio,template=data.load_validation(parent,family)
            for seed in range(2):
                target=OUT/'validation'/str(update)/family.replace(':','_')/str(seed)
                record=old.cached_record(target/'record.json',identity)
                if record is None:
                    if time.monotonic()>=deadline:return {'status':'INCOMPLETE','update':update,'family':family,'seed':seed}
                    attempt=geometry.refine(model,template,audio,seed,1.)
                    record=data.measure(attempt,template,reference,target,identity,parent['config']['validation_scales'])
                    print('hand-state validation',update,family,seed,record['ok'],flush=True)
                records.append(record)
        summary={**data.aggregate(records),'update':update,'snapshot_sha256':sha};summaries.append(summary);by_update[update]=records
        _atomic(OUT/'validation'/f'summary-{update}.json',summary)
    selected=choose(summaries);gates=qualify(selected,prepared['control'],by_update[selected['update']])
    result={'identity':frozen['identity'],'status':'VALIDATION_POSITIVE' if all(gates.values()) else 'VALIDATION_NEGATIVE',
        'selected':selected['update'],'snapshot_sha256':selected['snapshot_sha256'],'snapshots':summaries,'gates':gates,
        'control':prepared['control'],'release_eligible':False,'selection':frozen['config']['selection'],
        'scope':'six opened approved validation proxies; no machine admission or fresh confirmation claim'}
    path=OUT/'selection.json'
    if path.exists():assert json.loads(path.read_text())==result
    _atomic(path,result);return result


if __name__=='__main__':print(json.dumps({'train':train,'validate':validate}[sys.argv[1]](),indent=1))
