"""A fixed audio/workload ridge baseline for phrase budgets, not generated maps."""
import json
import math

import numpy as np
import torch

from eval import phrase_state as codec,joint_diverse_fit as corpus,joint_approved_fit as approved,joint_uniform_audio as uniform,joint_pilot as old,joint_ordered as ordered
from eval.expressive_manifest import freeze_run
from eval.joint_model import _torch_save
from eval.joint_phrase import ROOT,_atomic,_sha

OUT=ROOT/'experiments/phrase-budget-pilot-v1'
SPEC=ROOT/'docs/specs/2026-09-30-phrase-budget-pilot-design.md'


def features(audio,start,end,bpm,requested_rate,duration):
    if not all(math.isfinite(x) for x in (start,end,bpm,requested_rate,duration)) or not 0<=start<end<=duration or bpm<=0 or requested_rate<=0:
        raise ValueError('invalid serving feature inputs')
    descriptor=old.phrase_descriptor(audio,start,end,bpm,requested_rate*60/bpm);descriptor[-1]=requested_rate*60/bpm
    result=np.r_[descriptor,ordered.temporal_descriptor(audio,start,end,bpm),60/bpm,(end-start)/8,start/duration,end/duration].astype(np.float64)
    if result.shape!=(91,) or not np.isfinite(result).all():raise ValueError('invalid audio features')
    return result


def fit_ridge(x,y,weights,roles):
    if len(roles)!=len(x) or set(roles)!={'train'}:raise ValueError('ridge fitting is train-only')
    x=np.asarray(x,dtype=np.float64);y=np.asarray(y,dtype=np.float64);weights=np.asarray(weights,dtype=np.float64)
    if x.ndim!=2 or y.shape!=(len(x),3) or weights.shape!=(len(x),) or not np.isfinite(x).all() or not np.isfinite(y).all() or not np.isfinite(weights).all() or np.any(weights<=0):raise ValueError('invalid ridge data')
    weights=weights/weights.sum();mean=weights@x;sd=np.maximum(np.sqrt(weights@np.square(x-mean)),1e-6);target_mean=weights@y
    z=(x-mean)/sd;coef=np.linalg.solve(z.T@(weights[:,None]*z)+.01*np.eye(x.shape[1]),z.T@(weights[:,None]*(y-target_mean)))
    return {'mean':mean,'sd':sd,'target_mean':target_mean,'coef':coef,'ridge_penalty':.01}


def predict(model,x,requested_rate,seconds,arm):
    x=np.asarray(x,dtype=np.float64);seconds=np.asarray(seconds,dtype=np.float64)
    if arm not in ('ridge','constant') or not math.isfinite(requested_rate) or requested_rate<=0 or seconds.shape!=(len(x),) or np.any(seconds<=0) or not np.isfinite(seconds).all():raise ValueError('invalid budget prediction inputs')
    ratios=np.broadcast_to(model['target_mean'],(len(x),3)).copy()
    if arm=='ridge':ratios+=((x-model['mean'])/model['sd'])@model['coef']
    if not np.isfinite(ratios).all():raise ValueError('nonfinite predicted ratios')
    return np.floor(np.maximum(ratios,0)*requested_rate*seconds[:,None]+.5).astype(np.int64)


def rhythm(plans,bpm):
    rows=[]
    for p in plans:
        if len(p['counts'])!=3 or any(type(n) is not int or n<0 for n in p['counts']):raise ValueError('invalid integer budget')
        left,right,both=p['counts'];total=left+right+both;seconds=(p['end']-p['start'])*60/bpm
        if not math.isfinite(seconds) or seconds<=0:raise ValueError('invalid budget duration')
        rows.append([total/seconds,(left+both)/seconds,(right+both)/seconds,both/total if total else 0.,float(total==0)])
    return rows


def error(reference,generated,scales):
    blank=[[None]*5 for _ in reference]
    return old.errors({'rhythm':reference,'geometry':blank},{'rhythm':generated,'geometry':blank},scales)['rhythm']


def run():
    torch.set_num_threads(2);source_run=json.loads((corpus.OUT/'run.json').read_text());source_config=source_run['config']
    packet_report=json.loads((codec.OUT/'report.json').read_text());selection=json.loads((corpus.OUT/'selection.json').read_text())
    calibration=json.loads((approved.OUT/'prepared.json').read_text());scales=calibration['validation_scales'];requested=calibration['validation_rate']
    inputs={};train_x=[];train_y=[];weights=[];roles=[];families=[];validation={};train_count=0
    for family,item in sorted(source_config['sources'].items()):
        rp=codec.OUT/'packets'/(family.replace(':','_')+'.json');receipt=packet_report['receipts'][family]
        if receipt['role']!=item['role'] or _sha(rp)!=receipt['packet_sha256']:raise ValueError('phrase source or role changed')
        packet=json.loads(rp.read_text());meta=packet['metadata'];plans=packet['plans'];seconds=np.array([(p['end']-p['start'])*60/meta['bpm'] for p in plans])
        view_root=uniform.OUT/'audio'/family.replace(':','_') if family in source_config['uniform_views'] else approved.OUT/'audio'/family.replace(':','_')
        view_receipt=source_config['uniform_views'].get(family,source_config['validation_views'].get(family))
        if _sha(view_root/'features.pt')!=view_receipt['features_sha256']:raise ValueError('audio view changed')
        audio=torch.load(view_root/'features.pt',weights_only=False)
        rate=sum(a+b+2*c for a,b,c in (p['counts'] for p in plans))/(meta['duration_beats']*60/meta['bpm']) if item['role']=='train' else requested
        x=np.stack([features(audio,p['start'],p['end'],meta['bpm'],rate,meta['duration_beats']) for p in plans])
        inputs[family]={'role':item['role'],'packet_sha256':receipt['packet_sha256'],'view_sha256':view_receipt['features_sha256'],'requested_rate':rate}
        if item['role']=='train':
            train_count+=1;train_x.extend(x);train_y.extend(np.array([p['counts'] for p in plans])/(rate*seconds[:,None]));weights.extend([1/len(plans)]*len(plans));roles.extend(['train']*len(plans));families.extend([family]*len(plans))
        elif item['role']=='validation':
            if _sha(ROOT/item['payload'])!=item['payload_sha256']:raise ValueError('validation reference changed')
            data=torch.load(ROOT/item['payload'],weights_only=False);ref=old.profile(data['source'],data['audio'])['rhythm']
            assert rhythm(plans,meta['bpm'])==ref and error(ref,ref,scales)['mean']==0
            controls=[]
            for seed in range(2):
                path=corpus.OUT/'validation'/str(selection['selected'])/family.replace(':','_')/str(seed)/'record.json'
                rec=old.cached_record(path,source_run['identity']+':'+selection['snapshot_sha256']);assert rec is not None and rec['ok']
                controls.append({'record_sha256':_sha(path),'errors':rec['errors']['rhythm']})
            inputs[family]['reference_payload_sha256']=item['payload_sha256'];inputs[family]['controls']=controls
            validation[family]={'x':x,'seconds':seconds,'metadata':meta,'reference':ref,'plans':plans,'controls':controls}
        else:raise ValueError('unexpected source role')
    if train_count!=336 or set(validation)!=approved.VALIDATION or set(families)&set(validation) or 'fam:24227' in families:raise ValueError('planner split changed')
    frozen=freeze_run(OUT,{'source_identity':source_run['identity'],'packet_report_sha256':_sha(codec.OUT/'report.json'),
        'source_report_sha256':_sha(corpus.OUT/'report.json'),'code_sha256':_sha(__file__),'spec_sha256':_sha(SPEC),'inputs':inputs,
        'fixed_validation_calibration_sha256':_sha(approved.OUT/'prepared.json'),'ridge_penalty':.01,'feature_count':91,
        'dependencies':{p:_sha(ROOT/p) for p in ('eval/phrase_state.py','eval/joint_pilot.py','eval/joint_ordered.py')}})
    data={'x':np.array(train_x),'y':np.array(train_y),'weights':np.array(weights),'roles':roles,'families':families};model=fit_ridge(data['x'],data['y'],data['weights'],roles)
    for name,value in (('training.pt',data),('model.pt',model)):
        path=OUT/name
        if path.exists():np.testing.assert_equal(value,torch.load(path,weights_only=False))
        else:_torch_save(path,value)
    rows=[]
    for family,v in sorted(validation.items()):
        arms={}
        for arm in ('constant','ridge'):
            counts=predict(model,v['x'],requested,v['seconds'],arm)
            plans=[{**p,'counts':c.tolist()} for p,c in zip(v['plans'],counts)];profile=rhythm(plans,v['metadata']['bpm'])
            ref=np.array(v['reference']);got=np.array(profile);achieved=float(np.sum(counts[:,0]+counts[:,1]+2*counts[:,2])/v['seconds'].sum())
            arms[arm]={'plans':plans,'rhythm':profile,'errors':error(v['reference'],profile,scales),'achieved_rate':achieved,
                'workload_ratio':achieved/requested,'generated_rest_over_reference_notes':np.where((ref[:,4]==0)&(got[:,4]==1))[0].tolist(),
                'generated_notes_in_reference_rest':np.where((ref[:,4]==1)&(got[:,4]==0))[0].tolist(),
                'activity_std':float(got[:,0].std()),'reference_activity_std':float(ref[:,0].std()),
                'mean_double_share':float(got[:,3].mean()),'reference_mean_double_share':float(ref[:,3].mean())}
        controls=v['controls'];arms['joint_control']={'errors':{'mean':float(np.mean([r['errors']['mean'] for r in controls])),
            'components':np.mean([r['errors']['components'] for r in controls],axis=0).tolist()}}
        rows.append({'family':family,'requested_rate':requested,'arms':arms})
    aggregate={a:{'mean':float(np.mean([r['arms'][a]['errors']['mean'] for r in rows])),
        'components':np.mean([r['arms'][a]['errors']['components'] for r in rows],axis=0).tolist()} for a in ('constant','ridge','joint_control')}
    gates={'complete_panel':len(rows)==6,'workload_within_20percent':all(.8<=r['arms']['ridge']['workload_ratio']<=1.2 for r in rows)}
    for control in ('constant','joint_control'):
        a,b=aggregate['ridge'],aggregate[control]
        gates['mean_10percent_vs_'+control]=a['mean']<=.9*b['mean'] and a['mean']<b['mean']
        gates['components_vs_'+control]=all(x<=1.1*y+.01 for x,y in zip(a['components'],b['components']))
    result={'identity':frozen['identity'],'status':'PLANNING_SIGNAL_PRESENT' if all(gates.values()) else 'PLANNING_SIGNAL_NOT_ESTABLISHED',
        'gates':gates,'aggregate':aggregate,'families':rows,'train_families':train_count,'train_windows':len(train_x),
        'training_sha256':_sha(OUT/'training.pt'),'model_sha256':_sha(OUT/'model.pt'),
        'claim':'budget prediction proxy only; no note times, geometry, physical flow, QA admission or generated-map quality claim'}
    path=OUT/'report.json'
    if path.exists():assert json.loads(path.read_text())==result
    _atomic(path,result);return {k:v for k,v in result.items() if k!='families'}


if __name__=='__main__':print(json.dumps(run(),indent=1))
