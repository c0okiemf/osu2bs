"""Uniform common-band donor/query features with fixed source and moment inventories."""
from collections import Counter
import json
import time
from unittest.mock import patch

import numpy as np
import torch

from eval import joint_expanded as control,joint_quarantine as quarantine,joint_audio_view as view
from eval import joint_recovered_inventory as recovered,joint_compatible as compatible,joint_ordered as ordered,joint_pilot as old
from eval.expressive_manifest import freeze_run
from eval.joint_model import audio_features,_torch_save
from eval.joint_phrase import ROOT,_atomic,_sha
from eval.joint_export import read_chart
from eval.joint_continuity import describe

OUT=ROOT/'experiments/joint-phrase-v15/uniform-audio'
SPEC=ROOT/'docs/specs/2026-09-30-uniform-audio-frontend-design.md'


def source_inventory():
    records=json.loads((old.OUT/'run.json').read_text())['config']['records'];inventory=json.loads((recovered.OUT/'inventory.json').read_text())
    additions=json.loads((recovered.OUT/'run.json').read_text())['config']['new_sources'];sources={}
    for r,root in [(r,old.OUT) for r in records]+[(r['composite_record'],recovered.OUT) for r in additions]:
        family=r['family'];role=inventory['roles']['sources'][family]['future_role']
        if role not in ('train','development'):continue
        path=root/'data'/(family.replace(':','_')+'.pt');receipt=json.loads(path.with_suffix('.json').read_text())
        if _sha(path)!=receipt['sha256'] or _sha(r['sources']['audio']['path'])!=r['sources']['audio']['sha256']:raise ValueError('uniform audio input changed')
        entry={'role':role,'audio':r['sources']['audio'],'payload_sha256':receipt['sha256']}
        if role=='train':entry['source_bpm']=torch.load(path,weights_only=False,mmap=True)['source']['bpm']
        sources[family]=entry
    if Counter(r['role'] for r in sources.values())!={'train':336,'development':8}:raise ValueError('uniform audio role inventory changed')
    return sources


def rebuild(normalization,donors,sources,features):
    def entries(rows):
        result=[]
        for e in rows:
            if sources[e['family']]['role']!='train':raise ValueError('nontraining audio in donor/moment fitting')
            descriptor=old.phrase_descriptor(features[e['family']],e['start'],e['start']+8,sources[e['family']]['source_bpm'],e['descriptor'][-1])
            # Empty audio slices must not erase the already-correct workload coordinate.
            descriptor[-1]=e['descriptor'][-1]
            result.append({**e,'descriptor':descriptor})
        return result
    moments=entries(normalization['entries']);raw=np.stack([e['descriptor'] for e in moments]);mean=raw.mean(0);sd=np.maximum(raw.std(0),1e-6)
    np.testing.assert_equal(mean[-1],normalization['mean'][-1]);np.testing.assert_equal(sd[-1],normalization['sd'][-1])
    revised=entries(donors['entries']);x=np.stack([e['descriptor'] for e in revised])
    return {'entries':revised,'mean':mean,'sd':sd,'x':(x-mean)/sd}, {'keys':[(e['family'],e['start']) for e in moments],'descriptors':raw,'mean':mean,'sd':sd}


def freeze_experiment():
    parent,bank=control.prepare();report=json.loads((control.OUT/'report.json').read_text())
    if report['identity']!=parent['identity'] or not report['gates']['complete_panel']:raise ValueError('expanded control incomplete')
    return freeze_run(OUT,{'parent_identity':parent['identity'],'control_bank_receipt':bank,'source_inventory':source_inventory(),
        'code_sha256':_sha(__file__),'spec_sha256':_sha(SPEC),'render_recipe_sha256':_sha(view.__file__),
        'audio_recipe_sha256':_sha(ROOT/'eval/joint_model.py'),'excluded_train_families':parent['config']['excluded_train_families'],
        'control_report_sha256':_sha(control.OUT/'report.json'),'view_diagnostic_report_sha256':_sha(view.OUT/'report.json'),
        'normalization_bank_sha256':_sha(quarantine.OUT/'retrieval.pt'),
        'control_artifacts':{str(p.relative_to(ROOT)):_sha(p) for p in (control.OUT/'development').rglob('*')
            if p.is_file() and p.name in ('song.json','record.json','ExpertPlus.dat','Info.dat')}})


def prepare(deadline=None):
    frozen=freeze_experiment();receipts={}
    for family,source in sorted(frozen['config']['source_inventory'].items()):
        root=OUT/'audio'/family.replace(':','_');path=root/'features.pt';rp=root/'receipt.json'
        if not rp.exists():
            if deadline is not None and time.monotonic()>=deadline:raise TimeoutError('uniform audio preparation deadline')
            render=view.render_common(source['audio']['path'],root/'common14800.wav')
            features=audio_features(root/'common14800.wav');_torch_save(path,features)
            _atomic(rp,{'identity':frozen['identity'],'features_sha256':_sha(path),'render':render,'source_audio_sha256':source['audio']['sha256']})
            print('uniform audio prepared',family,source['role'],flush=True)
        r=json.loads(rp.read_text())
        if r['identity']!=frozen['identity'] or r['features_sha256']!=_sha(path) or r['render']['sha256']!=_sha(root/'common14800.wav') or r['source_audio_sha256']!=source['audio']['sha256']:raise ValueError('uniform audio cache changed')
        receipts[family]=r
    bp=OUT/'bank-receipt.json'
    if not bp.exists():
        normalization=torch.load(quarantine.OUT/'retrieval.pt',weights_only=False);donors=torch.load(control.OUT/'ordered.pt',weights_only=False)
        train={f:torch.load(OUT/'audio'/f.replace(':','_')/'features.pt',weights_only=False) for f,s in frozen['config']['source_inventory'].items() if s['role']=='train'}
        bank,moments=rebuild(normalization,donors,frozen['config']['source_inventory'],train)
        _torch_save(OUT/'retrieval.pt',bank);_torch_save(OUT/'moments.pt',moments)
        _atomic(bp,{'identity':frozen['identity'],'bank_sha256':_sha(OUT/'retrieval.pt'),'moments_sha256':_sha(OUT/'moments.pt'),
            'moment_entries':len(moments['keys']),'moment_families':len({k[0] for k in moments['keys']}),'donor_entries':len(bank['entries']),
            'donor_families':len({e['family'] for e in bank['entries']}),'audio_receipts':receipts})
    receipt=json.loads(bp.read_text())
    if receipt['identity']!=frozen['identity'] or receipt['bank_sha256']!=_sha(OUT/'retrieval.pt') or receipt['moments_sha256']!=_sha(OUT/'moments.pt') or receipt['audio_receipts']!=receipts:raise ValueError('uniform bank changed')
    return frozen,receipt


def decide(songs,frozen,controls,evidence):
    with patch.object(quarantine,'OUT',OUT):report=quarantine.decide(songs,frozen,controls,evidence)
    report['audio_frontend_ablation']=report.pop('quarantine_ablation')
    for d in report['audio_frontend_ablation'].values():
        d['mixed_frontend_mean']=d.pop('overlap_exposed_control_mean');d['uniform_frontend_mean']=d.pop('quarantined_bank_mean')
    report['arm_semantics']['ordered']='approved45 with uniform14800Hz donor/query preprocessing and unchanged literal retrieval'
    report['quarantine_contract']['similarity_normalizers']='same315-family/window fitting inventory, audio moments rebuilt uniformly; workload moments exact'
    report['audio_frontend']={'fit_source_views':336,'query_views':8,'validation_views':0,'measurement_audio':'original frozen audio features',
        'timing_authority':'original native audio metadata and deployment adapter; rendered duration never used for charts'}
    report['budget']='six new attempts per family; cached v14 approved45 controls; uniform audio/moments only, no model fit'
    _atomic(OUT/'report.json',report);return report


def run(deadline_seconds=2700):
    deadline=time.monotonic()+min(deadline_seconds,2700);torch.set_num_threads(2)
    try:frozen,receipt=prepare(deadline)
    except TimeoutError:return {'status':'INCOMPLETE','stage':'audio preparation'}
    bank=torch.load(OUT/'retrieval.pt',weights_only=False);previous_bank=torch.load(control.OUT/'ordered.pt',weights_only=False)
    scales=old.evaluation_freeze()['config']['scales'];kit=old.machine_tools();songs=[];controls={};evidence={}
    for family,item in ordered.inputs().items():
        key=family.replace(':','_');dest=OUT/'development'/key;root=control.OUT/'development'/key
        prior=json.loads((root/'song.json').read_text());controls[family]=prior;records=[];audits=[]
        query_audio=torch.load(OUT/'audio'/key/'features.pt',weights_only=False)
        for seed in range(6):
            target=dest/'ordered'/str(seed);rec=old.cached_record(target/'record.json',frozen['identity'])
            if rec is None:
                if time.monotonic()>=deadline:return {'status':'INCOMPLETE','family':family,'next_seed':seed,'families':len(songs)}
                started=time.monotonic();attempt=compatible.retrieve(item['source'],query_audio,bank,item['rate'],seed);attempt['elapsed_s']=time.monotonic()-started
                # Measurement deliberately receives ORIGINAL audio, not the query view.
                rec=ordered.measure(attempt,item['data'],target,frozen['identity'],scales,kit,root/'ordered'/str(seed)/'record.json')
                print(f'uniform frontend {family} s{seed}: {rec["ok"]} {rec.get("machine",{}).get("verdict")}',flush=True)
            audit=None
            if rec['ok']:
                src=read_chart(target/'ExpertPlus.dat',target/'Info.dat');src['duration_beats']=item['source']['duration_beats']
                audit=compatible.audit_ledger(src,rec['donors'],bank)
                if audit['compressed'] or audit['unknown'] or audit['nonterminal_unknown_exits']:raise ValueError('native context contract failed')
            records.append(rec);audits.append(audit)
        chosen=ordered.select(records,item['b0']);song={**prior,'identity':frozen['identity'],'arms':{'retrieval':prior['arms']['retrieval'],'ordered':chosen}}
        _atomic(dest/'song.json',song);songs.append(song)
        before=prior['arms']['ordered'];pp=old.PILOT/'development'/key/'b0' if before['fallback'] else root/'ordered'/str(before['selected_seed'])
        previous=read_chart(pp/'ExpertPlus.dat',pp/'Info.dat');previous['duration_beats']=item['source']['duration_beats']
        cp=old.PILOT/'development'/key/'b0' if chosen['fallback'] else dest/'ordered'/str(chosen['selected_seed'])
        evidence[family]={'attempts':audits,'selected_seed':chosen['selected_seed'],
            'mixed_frontend_control_audit':None if before['fallback'] else compatible.audit_ledger(previous,before['selected']['donors'],previous_bank),
            'continuity_control':describe(previous),'continuity_candidate':describe(read_chart(cp/'ExpertPlus.dat',cp/'Info.dat'))}
    prepare();return decide(songs,frozen,controls,evidence)


if __name__=='__main__':print(json.dumps(run(),indent=1))
