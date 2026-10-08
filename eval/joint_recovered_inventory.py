"""Prepare recovered approved audio and freeze leakage-aware future source roles."""
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from eval import joint_source_recovery as recovery,joint_audio_identity as identity,joint_pilot as old
from eval.corpus import _UF,_fingerprint,FP_CORR
from eval.expressive_manifest import freeze_run
from eval.joint_model import audio_features,_torch_save
from eval.joint_phrase import ROOT,_atomic,_sha,verify_sources

OUT=ROOT/'experiments/joint-recovered-inventory-v1'
SPEC=ROOT/'docs/specs/2026-09-30-recovered-inventory-design.md'
SALT='joint-recovered-approved-validation-2026-09-30:'


def assign_roles(old_roles,new_families,edges,prior_quarantine):
    new=set(new_families);names=sorted(set(old_roles)|new)
    if set(old_roles)&new:raise ValueError('recovered source already has an old role')
    index={f:i for i,f in enumerate(names)};uf=_UF(len(names))
    for a,b in edges:uf.union(index[a],index[b])
    components={}
    for f in names:components.setdefault(uf.find(index[f]),[]).append(f)
    groups=[sorted(g) for g in components.values()];fresh=[g for g in groups if all(f in new for f in g)]
    reserved={g[0] for g in sorted(fresh,key=lambda g:hashlib.sha256((SALT+g[0]).encode()).hexdigest())[:5]}
    priority={'train':0,'validation':1,'development':2};quarantine={r:set(fs) for r,fs in prior_quarantine.items()};source_roles={};out=[]
    for g in sorted(groups):
        inherited=[old_roles[f] for f in g if f in old_roles]
        role=max(inherited,key=priority.get) if inherited else 'validation' if g[0] in reserved else 'train'
        out.append({'group':g[0],'families':g,'role':role,'old_role_inherited':bool(inherited)})
        for f in g:
            previous=old_roles.get(f)
            if previous is not None and priority[previous]<priority[role]:quarantine.setdefault(previous,set()).add(f)
            excluded=previous is not None and f in quarantine.get(previous,set())
            source_roles[f]={'old_role':previous,'component_role':role,'future_role':None if excluded else role,'quarantined':excluded}
    return {'groups':out,'sources':source_roles,'quarantine':{r:sorted(fs) for r,fs in quarantine.items()},'new_only_validation_groups':sorted(reserved)}


def prepare():
    parent=old.freeze();r=json.loads((recovery.OUT/'report.json').read_text());new=[x for x in r['families'] if x['status']=='recovered_supported']
    prior=json.loads((identity.OUT/'run.json').read_text());prior_report=json.loads((identity.OUT/'report.json').read_text());q=json.loads((identity.OUT/'quarantine.json').read_text())
    if prior_report['identity']!=prior['identity'] or q['report_sha256']!=_sha(identity.OUT/'report.json'):raise ValueError('prior overlap evidence changed')
    if len(new)!=26:raise ValueError('recovery inventory changed')
    for row in new:
        verify_sources(row['approved_record']);verify_sources(row['composite_record'])
        if _sha(recovery.OUT/'sources'/(row['family'].replace(':','_')+'.json'))!=row['source_sha256']:raise ValueError('recovered source changed')
    for f,receipt in prior['config']['inputs'].items():
        path=old.OUT/'data'/(f.replace(':','_')+'.pt')
        if _sha(path)!=receipt['prepared_sha256'] or _sha(path.with_suffix('.json'))!=receipt['receipt_sha256']:raise ValueError('old prepared data changed')
        verify_sources({'sources':receipt['sources']})
    frozen=freeze_run(OUT,{'model_identity':parent['identity'],'recovery_report_sha256':_sha(recovery.OUT/'report.json'),
        'old_audio_report_sha256':_sha(identity.OUT/'report.json'),'old_quarantine_sha256':_sha(identity.OUT/'quarantine.json'),
        'new_sources':new,'old_inputs':prior['config']['inputs'],'code_sha256':_sha(__file__),'spec_sha256':_sha(SPEC),
        'dependencies':{p:_sha(ROOT/p) for p in ('eval/joint_audio_identity.py','eval/joint_model.py','eval/corpus.py')},
        'role_reservation_salt':SALT,'new_only_validation_count':5})
    receipts={}
    for row in sorted(new,key=lambda x:x['family']):
        key=row['family'].replace(':','_');path=OUT/'data'/(key+'.pt');rp=path.with_suffix('.json')
        if not rp.exists():
            source=json.loads((recovery.OUT/'sources'/(key+'.json')).read_text());audio=audio_features(row['composite_record']['sources']['audio']['path'])
            _torch_save(path,{'family':row['family'],'source':source,'audio':audio})
            _atomic(rp,{'identity':frozen['identity'],'sha256':_sha(path),'source_sha256':row['source_sha256']})
            print('prepared recovered audio',row['family'],flush=True)
        receipt=json.loads(rp.read_text())
        if receipt['identity']!=frozen['identity'] or receipt['sha256']!=_sha(path) or receipt['source_sha256']!=row['source_sha256']:raise ValueError('recovered payload changed')
        receipts[row['family']]=receipt
    return frozen,parent,new,prior_report,q,receipts


def run():
    torch.set_num_threads(2);frozen,parent,new,prior_report,q,receipts=prepare();envelopes={};audio_paths={}
    old_records={r['family']:r for r in parent['config']['records']};new_records={r['family']:r for r in new}
    for f in sorted(set(old_records)|set(new_records)):
        root=old.OUT if f in old_records else OUT;data=torch.load(root/'data'/(f.replace(':','_')+'.pt'),weights_only=False,mmap=True);a=data['audio']
        envelopes[f]=np.interp(np.arange(0,min(90.,a['duration_s']),.02),a['times'],a['rms'])
        audio_paths[f]=old_records[f]['sources']['audio']['path'] if f in old_records else new_records[f]['composite_record']['sources']['audio']['path']
        del data
    new_names=sorted(new_records);pairs=[(a,b) for a in new_names for b in sorted(old_records)]+[(a,b) for i,a in enumerate(new_names) for b in new_names[i+1:]]
    assert len(pairs)==9581
    scores=[];flagged=[];fingerprints={}
    for a,b in pairs:
        peak=identity.peak_correlation(envelopes[a],envelopes[b]);row={'families':[a,b],**peak};scores.append(row)
        if peak['correlation'] is None or peak['correlation']<FP_CORR:continue
        for family in (a,b):
            if family not in fingerprints:fingerprints[family]=_fingerprint(Path(audio_paths[family]))
        native=identity.peak_correlation(fingerprints[a],fingerprints[b]);confirmed=native['correlation'] is not None and native['correlation']>=FP_CORR
        flagged.append({**row,'independent_original_decode':native,'corroborated_audio_overlap':confirmed})
        print('recovered overlap',a,b,peak,native,flush=True)
    edges=[r['families'] for r in prior_report['flagged_pairs']+flagged if r['corroborated_audio_overlap']]
    roles=assign_roles({f:r['fit_role'] for f,r in old_records.items()},new_names,edges,q['exclude_from_future_fit_roles'])
    report={'identity':frozen['identity'],'pairs_screened':len(scores),'scores':scores,'flagged_pairs':flagged,'roles':roles,
        'new_role_counts':dict(Counter(roles['sources'][f]['future_role'] for f in new_names)),
        'claim':'recovered source/audio integrity and future role inventory only; old models/B0/QA retain past exposure; no quality or fresh-test claim'}
    _atomic(OUT/'report.json',report)
    _atomic(OUT/'inventory.json',{'identity':frozen['identity'],'report_sha256':_sha(OUT/'report.json'),'roles':roles,'new_payload_receipts':receipts,
                                'role_scope':'future experiments only; preserve component grouping in sampling and evaluation'})
    return {k:v for k,v in report.items() if k not in ('scores','roles')}


if __name__=='__main__':print(json.dumps(run(),indent=1))
