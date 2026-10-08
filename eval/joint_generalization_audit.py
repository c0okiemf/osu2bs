"""Read-only teacher prediction audit: source generalization versus recurrent history."""
from collections import defaultdict
import json

import numpy as np
import torch
import torch.nn.functional as F

from eval import joint_approved_fit as fit
from eval.expressive_manifest import freeze_run
from eval.joint_model import KEYS,EMPTY,JointModel,actions,action_notes,context,source_rate
from eval.joint_phrase import ROOT,_atomic,_sha

OUT=ROOT/'experiments/joint-generalization-audit-v1'
PLAN=ROOT/'docs/specs/2026-09-30-joint-generalization-audit.md'


def spans(length,cropped):
    starts=[0]+list(range(96,length-32,96)) if cropped else [0]
    return [(s,min(s+128,length) if cropped else length,32 if s else 0) for s in starts]


def evaluation_inputs(source,audio,rate,njs,environment):
    rows=actions(source)
    if action_notes(rows)!=list(map(tuple,source['notes'])):raise ValueError('evaluation action roundtrip changed')
    ctx=np.stack([context(audio,r['cursor'],source['bpm'],rate,njs,environment['walls'],environment['bombs']) for r in rows])
    result={'context':torch.from_numpy(ctx),'prev':torch.tensor([r['prev_slots'] for r in rows]),
        'kind':torch.tensor([r['prev_kind'] for r in rows]),'prev_gap':torch.tensor([r['prev_gap'] for r in rows],dtype=torch.float32),
        'gap':torch.tensor([r['gap'] for r in rows]),'residual':torch.tensor([r['residual'] for r in rows],dtype=torch.float32),
        'count':torch.tensor([r['count'] for r in rows]),'slots':torch.tensor([r['slots'] for r in rows])}
    return result


def measure(model,data,cropped):
    totals=defaultdict(lambda:{'loss_sum':0.,'correct':0,'targets':0})
    with torch.no_grad():
        for start,end,warmup in spans(len(data['gap']),cropped):
            batch={k:v[start:end][None] for k,v in data.items()};out,_=model(batch)
            valid=torch.ones((1,end-start),dtype=torch.bool);valid[:,:warmup]=False
            event=valid&(batch['gap']>0);positive=valid&(batch['gap']>1);slots=event[...,None]&(batch['slots']!=EMPTY)
            for name,mask in (('gap',valid),('count',event),('slots',slots)):
                pred=out[name][mask];target=batch[name][mask];r=totals[name]
                if not len(target):continue
                r['loss_sum']+=float(F.cross_entropy(pred,target,reduction='sum'));r['targets']+=len(target)
                r['correct']+=int((pred.argmax(-1)==target).sum())
            pred=out['residual'].gather(-1,batch['gap'][...,None]).squeeze(-1);r=totals['residual']
            r['targets']+=int(positive.sum())
            if positive.any():r['loss_sum']+=float(F.smooth_l1_loss(pred[positive],batch['residual'][positive],reduction='sum'))
    return {k:{**r,'loss':r['loss_sum']/r['targets'] if r['targets'] else None,
        'accuracy':r['correct']/r['targets'] if r['targets'] and k!='residual' else None} for k,r in totals.items()}


def run():
    torch.set_num_threads(2);parent,prepared=fit.prepare();selection=json.loads((fit.OUT/'selection.json').read_text())
    snapshot=fit.OUT/f"snapshot-{selection['selected']}.pt"
    if _sha(snapshot)!=selection['snapshot_sha256']:raise ValueError('selected model changed')
    frozen=freeze_run(OUT,{'parent_identity':parent['identity'],'prepared_sha256':_sha(fit.OUT/'prepared.json'),
        'selection_sha256':_sha(fit.OUT/'selection.json'),'snapshot_sha256':_sha(snapshot),'code_sha256':_sha(__file__),'plan_sha256':_sha(PLAN)})
    model=JointModel().eval();model.load_state_dict(torch.load(snapshot,map_location='cpu',weights_only=True));rows=[]
    for family,item in sorted(parent['config']['sources'].items()):
        data=torch.load(ROOT/item['payload'],weights_only=False);source=data['source'];audio=fit.load_view(family,parent)
        conditions={'authored':(source_rate(source),source['authored']['njs'],source)}
        if item['role']=='validation':conditions['generation']=(prepared['validation_rate'],18,{'walls':[],'bombs':[]})
        for name,(rate,njs,environment) in conditions.items():
            tensors=evaluation_inputs(source,audio,rate,njs,environment)
            if item['role']=='train':
                cached=torch.load(fit.OUT/'data'/(family.replace(':','_')+'.pt'),weights_only=False)
                for key in KEYS:assert torch.equal(cached[key],tensors[key]),(family,key)
            result={'family':family,'role':item['role'],'conditioning':name,'continuous':measure(model,tensors,False),'cropped':measure(model,tensors,True)}
            for axis in result['continuous']:assert result['continuous'][axis]['targets']==result['cropped'][axis]['targets']
            rows.append(result)
    aggregate={}
    for role,condition in (('train','authored'),('validation','authored'),('validation','generation')):
        subset=[r for r in rows if r['role']==role and r['conditioning']==condition];groups={}
        for mode in ('continuous','cropped'):
            groups[mode]={}
            for axis in ('gap','count','slots','residual'):
                values=[r[mode][axis] for r in subset];n=sum(v['targets'] for v in values)
                groups[mode][axis]={'families':len(values),'targets':n,'family_mean_loss':float(np.mean([v['loss'] for v in values])),
                    'weighted_loss':sum(v['loss_sum'] for v in values)/n,'weighted_accuracy':sum(v['correct'] for v in values)/n if axis!='residual' else None}
        aggregate[role+'_'+condition]=groups
    assert _sha(snapshot)==frozen['config']['snapshot_sha256'];fit.freeze_experiment()
    result={'identity':frozen['identity'],'aggregate':aggregate,'families':rows,'claim':'teacher-only diagnostic; true-history validation is not serving; no fit or generated quality claim'}
    _atomic(OUT/'report.json',result);return {k:v for k,v in result.items() if k!='families'}


if __name__=='__main__':print(json.dumps(run(),indent=1))
