"""Train-only masked geometry prediction; no map selection or generation."""
import json
import numpy as np
import torch

from eval import phrase_budget as budget,joint_diverse_fit as corpus,joint_approved_fit as approved,joint_uniform_audio as uniform,joint_pilot as old
from eval.expressive_manifest import freeze_run
from eval.joint_model import _torch_save
from eval.joint_phrase import ROOT,_sha,_atomic

OUT=ROOT/'experiments/phrase-geometry-pilot-v1'
SPEC=ROOT/'docs/specs/2026-09-30-phrase-geometry-pilot-design.md'


def fit(x,y,families,roles):
    x=np.asarray(x,float);y=np.asarray(y,float);families=np.asarray(families)
    if len(roles)!=len(x) or set(roles)!={'train'}:raise ValueError('geometry fitting is train-only')
    if x.ndim!=2 or y.shape!=(len(x),5) or families.shape!=(len(x),) or not np.isfinite(x).all() or np.isinf(y).any():raise ValueError('invalid geometry training arrays')
    heads=[]
    for k in range(5):
        valid=np.isfinite(y[:,k]);a=x[valid];b=y[valid,k];fs=families[valid]
        if not len(a):raise ValueError('unsupported training component')
        _,ids,counts=np.unique(fs,return_inverse=True,return_counts=True)
        w=1/counts[ids];w=w/w.sum();mean=w@a;sd=np.maximum(np.sqrt(w@np.square(a-mean)),1e-6);target_mean=float(w@b)
        z=(a-mean)/sd;coef=np.linalg.solve(z.T@(w[:,None]*z)+.01*np.eye(x.shape[1]),z.T@(w*(b-target_mean)))
        heads.append({'mean':mean,'sd':sd,'target_mean':target_mean,'coef':coef,'supported_families':len(counts),'supported_windows':int(valid.sum())})
    return heads


def predict(heads,x,arm):
    if arm not in ('constant','ridge'):raise ValueError('unknown geometry predictor')
    x=np.asarray(x,float)
    if not np.isfinite(x).all():raise ValueError('nonfinite geometry features')
    result=np.stack([np.full(len(x),h['target_mean'])+(((x-h['mean'])/h['sd'])@h['coef'] if arm=='ridge' else 0) for h in heads],axis=1)
    if not np.isfinite(result).all():raise ValueError('nonfinite geometry prediction')
    result=np.maximum(result,0);result[:,[0,3]]=np.minimum(result[:,[0,3]],1.)
    return result


def features(base_x,seconds,rate,model):
    seconds=np.asarray(seconds,float)
    counts=budget.predict(model,base_x,rate,seconds,'ridge')
    return np.column_stack((base_x,counts/(rate*seconds[:,None])))


def error(reference,predicted,scales):
    blank=[[None]*5 for _ in reference]
    return old.errors({'rhythm':blank,'geometry':reference},{'rhythm':blank,'geometry':predicted},scales)['geometry']


def run():
    torch.set_num_threads(2);br=json.loads((budget.OUT/'report.json').read_text());bc=json.loads((budget.OUT/'run.json').read_text())['config']
    if br['status']!='PLANNING_SIGNAL_PRESENT' or _sha(budget.OUT/'training.pt')!=br['training_sha256'] or _sha(budget.OUT/'model.pt')!=br['model_sha256']:raise ValueError('budget predictor changed')
    base=torch.load(budget.OUT/'training.pt',weights_only=False);model=torch.load(budget.OUT/'model.pt',weights_only=False)
    source_run=json.loads((corpus.OUT/'run.json').read_text());sources=source_run['config']['sources'];selection=json.loads((corpus.OUT/'selection.json').read_text())
    calibration=json.loads((approved.OUT/'prepared.json').read_text());scales=calibration['validation_scales'];validation_rate=calibration['validation_rate']
    train_x=[];train_y=[];families=[];roles=[];validation={};inputs={};base_families=np.asarray(base['families'])
    for family,item in sorted(sources.items()):
        if item['role'] not in ('train','validation') or item['role']!=bc['inputs'][family]['role']:raise ValueError('source role changed')
        path=ROOT/item['payload']
        if _sha(path)!=item['payload_sha256']:raise ValueError('geometry source payload changed')
        data=torch.load(path,weights_only=False);source=data['source'];bpm=source['bpm'];duration=source['duration_beats']
        windows=[(float(s),min(float(s)+8,duration)) for s in np.arange(0.,duration,8.)]
        seconds=np.array([(e-s)*60/bpm for s,e in windows]);profile=old.profile(source,data['audio'])['geometry']
        if len(profile)!=len(windows):raise ValueError('geometry/window mismatch')
        rate=bc['inputs'][family]['requested_rate'];inputs[family]={'role':item['role'],'payload_sha256':item['payload_sha256'],'requested_rate':rate}
        if item['role']=='train':
            ids=np.flatnonzero(base_families==family)
            if len(ids)!=len(windows) or any(base['roles'][i]!='train' for i in ids):raise ValueError('base feature/window mismatch')
            x=features(base['x'][ids],seconds,rate,model);train_x.extend(x);train_y.extend(profile);families.extend([family]*len(x));roles.extend(['train']*len(x))
        else:
            if rate!=validation_rate:raise ValueError('validation request changed')
            root=approved.OUT/'audio'/family.replace(':','_');sha=bc['inputs'][family]['view_sha256']
            if _sha(root/'features.pt')!=sha:raise ValueError('validation audio changed')
            audio=torch.load(root/'features.pt',weights_only=False)
            base_x=np.stack([budget.features(audio,s,e,bpm,rate,duration) for s,e in windows]);x=features(base_x,seconds,rate,model)
            controls=[]
            for seed in range(2):
                path=corpus.OUT/'validation'/str(selection['selected'])/family.replace(':','_')/str(seed)/'record.json'
                if _sha(path)!=bc['inputs'][family]['controls'][seed]['record_sha256']:raise ValueError('validation control changed')
                r=old.cached_record(path,source_run['identity']+':'+selection['snapshot_sha256']);assert r is not None and r['ok'];controls.append(r['errors']['geometry'])
            inputs[family].update(view_sha256=sha,controls=bc['inputs'][family]['controls'])
            validation[family]={'x':x,'reference':profile,'controls':controls}
    if len(set(families))!=336 or set(validation)!=approved.VALIDATION or set(families)&set(validation) or 'fam:24227' in families:raise ValueError('geometry split changed')
    frozen=freeze_run(OUT,{'code_sha256':_sha(__file__),'spec_sha256':_sha(SPEC),'source_identity':source_run['identity'],
        'inputs':inputs,'budget_report_sha256':_sha(budget.OUT/'report.json'),'base_feature_sha256':br['training_sha256'],
        'calibration_sha256':_sha(approved.OUT/'prepared.json'),'feature_count':94,'ridge_penalty':.01,
        'pool_report_sha256':_sha(ROOT/'experiments/phrase-native-pool-feasibility-v1/report.json'),
        'dependencies':{p:_sha(ROOT/p) for p in ('eval/phrase_budget.py','eval/joint_pilot.py','eval/expression_profile.py')}})
    training={'x':np.array(train_x),'y':np.array(train_y,float),'families':families,'roles':roles};heads=fit(training['x'],training['y'],families,roles)
    for name,value in (('training.pt',training),('model.pt',heads)):
        path=OUT/name
        if path.exists():np.testing.assert_equal(value,torch.load(path,weights_only=False))
        else:_torch_save(path,value)
    rows=[]
    for family,v in sorted(validation.items()):
        arms={}
        for arm in ('constant','ridge'):
            profile=predict(heads,v['x'],arm).tolist();arms[arm]={'profile':profile,'errors':error(v['reference'],profile,scales)}
        arms['joint_control']={'errors':{'mean':float(np.mean([e['mean'] for e in v['controls']])),
            'components':np.mean([e['components'] for e in v['controls']],axis=0).tolist()}}
        rows.append({'family':family,'arms':arms})
    aggregate={arm:{'mean':float(np.mean([r['arms'][arm]['errors']['mean'] for r in rows])),
        'components':np.mean([r['arms'][arm]['errors']['components'] for r in rows],axis=0).tolist()} for arm in ('constant','ridge','joint_control')}
    gates={'complete_panel':len(rows)==6}
    for control in ('constant','joint_control'):
        a,b=aggregate['ridge'],aggregate[control];gates['mean_10percent_vs_'+control]=a['mean']<=.9*b['mean'] and a['mean']<b['mean']
        gates['components_vs_'+control]=all(x<=1.1*y+.01 for x,y in zip(a['components'],b['components']))
    result={'identity':frozen['identity'],'status':'GEOMETRY_SIGNAL_PRESENT' if all(gates.values()) else 'GEOMETRY_SIGNAL_NOT_ESTABLISHED',
        'gates':gates,'aggregate':aggregate,'families':rows,'training_sha256':_sha(OUT/'training.pt'),'model_sha256':_sha(OUT/'model.pt'),
        'training_support':[{'families':h['supported_families'],'windows':h['supported_windows']} for h in heads],
        'claim':'held-out geometry prediction proxy only; no usable selector, generated quality or production claim'}
    path=OUT/'report.json'
    if path.exists():assert json.loads(path.read_text())==result
    _atomic(path,result);return {k:v for k,v in result.items() if k!='families'}


if __name__=='__main__':print(json.dumps(run(),indent=1))
