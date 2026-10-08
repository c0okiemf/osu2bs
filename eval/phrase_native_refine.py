"""Typed geometry refinement of an already-verified learned rhythm plan."""
import json
from pathlib import Path
import tempfile
import time
from unittest.mock import patch

import numpy as np
import torch

from eval import phrase_native as native,phrase_native_likelihood as scorer,joint_refine as typed,joint_ordered as ordered,joint_pilot as old
from eval import joint_uniform_audio as uniform,joint_approved_fit as approved
from eval.expressive_manifest import freeze_run
from eval.joint_model import JointModel
from eval.joint_phrase import ROOT,_sha,_atomic
from eval.joint_export import read_chart
from eval.joint_continuity import describe

OUT=ROOT/'experiments/phrase-native-refinement-v1'
SPEC=ROOT/'docs/specs/2026-09-30-budget-native-refinement-design.md'
TEMPERATURES=(.85,1.,1.15,.85,1.,1.15)


def inputs():
    result={}
    for family,item in ordered.inputs().items():
        root=native.OUT/'development'/family.replace(':','_');prior=json.loads((root/'song.json').read_text());base=prior['arms']['ordered']
        if base['fallback'] or not base['selected']['machine']['admitted']:raise ValueError('admitted native template required')
        path=root/'ordered'/str(base['selected_seed']);r=old.cached_record(path/'record.json',prior['identity']);assert r==base['selected']
        template=read_chart(path/'ExpertPlus.dat',path/'Info.dat');template['duration_beats']=item['source']['duration_beats']
        result[family]={**item,'prior':prior,'template':template,'template_record':r,'template_seed':base['selected_seed'],'template_path':path}
    return result


def select(records,template,template_seed):
    seed=next((i for i,r in enumerate(records) if r['ok'] and r.get('machine',{}).get('admitted')),None)
    return {'selected_seed':seed,'fallback':False,'refinement_fallback':seed is None,'template_seed':template_seed,
            'selected':template if seed is None else records[seed],'attempts':records}


def verify_export(template,actual,record,baseline):
    if not record.get('signature_ok') or typed.signature(template)!=typed.signature(actual):raise ValueError('typed template changed')
    for key in ('bombs','walls'):assert list(map(tuple,actual[key]))==list(map(tuple,template[key])),key
    assert actual['bpm']==template['bpm'] and actual['settings']['njs']==18 and actual['settings']['offset_beats']==0
    assert record['profile']['rhythm']==baseline['profile']['rhythm']
    np.testing.assert_array_equal(np.isfinite(np.array(record['profile']['geometry'],float)),np.isfinite(np.array(baseline['profile']['geometry'],float)))


def freeze():
    parent=scorer.freeze();templates=inputs()
    return freeze_run(OUT,{'parent_identity':parent['config']['native_identity'],'frozen_scorer_inputs':parent['identity'],
        'checkpoint':parent['config']['checkpoint'],'checkpoint_sha256':parent['config']['checkpoint_sha256'],
        'templates':{f:{'seed':r['template_seed'],'record_sha256':_sha(r['template_path']/'record.json'),
            'artifacts':r['template_record']['artifacts']} for f,r in templates.items()},
        'native_artifacts':parent['config']['artifacts'],'query_views':parent['config']['query_views'],'protected':parent['config']['protected'],
        'code_sha256':_sha(__file__),'spec_sha256':_sha(SPEC),'temperatures':TEMPERATURES,
        'arc_origin_report_sha256':_sha(ROOT/'experiments/phrase-native-arc-origin-v1/report.json'),
        'dependencies':{p:_sha(ROOT/p) for p in ('eval/joint_refine.py','eval/joint_model.py','eval/joint_ordered.py',
            'eval/joint_pilot.py','eval/joint_export.py','eval/joint_continuity.py','eval/joint_approved_fit.py')}})


def evidence(item,records,selected,root):
    gaps=[]
    for seed,r in enumerate(records):
        if not r['ok']:gaps.append(None);continue
        source=read_chart(root/'ordered'/str(seed)/'ExpertPlus.dat',root/'ordered'/str(seed)/'Info.dat')
        verify_export(item['template'],source,r,item['template_record']);gaps.append(approved.gap_diagnostic(source))
    chosen=item['template_path'] if selected['refinement_fallback'] else root/'ordered'/str(selected['selected_seed'])
    actual=read_chart(chosen/'ExpertPlus.dat',chosen/'Info.dat')
    return {'attempt_gaps':gaps,'continuity_template':describe(item['template']),'continuity_selected':describe(actual)}


def decide(songs,frozen,proofs):
    with tempfile.TemporaryDirectory(prefix='osu2bs-native-refinement-report-') as tmp:
        with patch.object(ordered,'OUT',Path(tmp)):report=ordered.decide(songs,frozen)
    refined=sum(not s['arms']['ordered']['refinement_fallback'] for s in songs)
    report['gates']['refinement_six_admitted_families']=refined>=6
    report['status']='DEVELOPMENT_POSITIVE' if all(report['gates'].values()) else 'DEVELOPMENT_NEGATIVE'
    report['refinement']={s['family']:{'native_template_seed':s['arms']['ordered']['template_seed'],
        'refinement_seed':s['arms']['ordered']['selected_seed'],'native_fallback':s['arms']['ordered']['refinement_fallback']} for s in songs}
    report['native_paired_changes']={axis:{s['family']:s['arms']['ordered']['selected']['errors'][axis]['mean']-
        json.loads((native.OUT/'development'/s['family'].replace(':','_')/'song.json').read_text())['arms']['ordered']['selected']['errors'][axis]['mean']
        for s in songs} for axis in ('rhythm','geometry')}
    report['geometry_evidence']=proofs
    report['construction_cost']={'cached_native_proposals':48,'new_refinement_proposals':48,'total_proposals':96,'families_with_admitted_refinement':refined}
    report['budget']='six cached native proposals plus six new typed-geometry proposals per song; frozen v19 model; no fit'
    report['fallback_semantics']='fallback means production B0 in inherited gates; refinement native_fallback retains an already admitted non-B0 template'
    saved=OUT/'report.json'
    if saved.exists():assert json.loads(saved.read_text())==report
    _atomic(saved,report);return report


def run():
    torch.set_num_threads(2);deadline=time.monotonic()+2700;frozen=freeze();model=JointModel().eval()
    model.load_state_dict(torch.load(ROOT/frozen['config']['checkpoint'],map_location='cpu',weights_only=True))
    kit=old.machine_tools();scales=old.evaluation_freeze()['config']['scales'];songs=[];proofs={}
    for family,item in inputs().items():
        key=family.replace(':','_');root=OUT/'development'/key;records=[]
        audio=torch.load(uniform.OUT/'audio'/key/'features.pt',weights_only=False)
        for seed,temperature in enumerate(TEMPERATURES):
            target=root/'ordered'/str(seed);record=old.cached_record(target/'record.json',frozen['identity'])
            if record is None:
                if time.monotonic()>=deadline:return {'status':'INCOMPLETE','family':family,'seed':seed}
                attempt=typed.refine(model,item['template'],audio,seed,temperature)
                attempt['signature_ok']=typed.signature(attempt['source'])==typed.signature(item['template'])
                if attempt['ok'] and not attempt['signature_ok']:raise ValueError('refiner changed signature')
                attempt['template_artifacts']=item['template_record']['artifacts']
                record=ordered.measure(attempt,item['data'],target,frozen['identity'],scales,kit,item['template_path']/'record.json')
                print('native refinement',family,seed,record['ok'],record.get('machine',{}).get('verdict',record['reason']),flush=True)
            if record['ok']:verify_export(item['template'],read_chart(target/'ExpertPlus.dat',target/'Info.dat'),record,item['template_record'])
            records.append(record)
        chosen=select(records,item['template_record'],item['template_seed'])
        song={**item['prior'],'identity':frozen['identity'],'arms':{'retrieval':item['prior']['arms']['retrieval'],'ordered':chosen}}
        _atomic(root/'song.json',song);songs.append(song);proofs[family]=evidence(item,records,chosen,root)
    freeze();return decide(songs,frozen,proofs)


if __name__=='__main__':print(json.dumps(run(),indent=1))
