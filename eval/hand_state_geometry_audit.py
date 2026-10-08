"""Rebuild hand-state data and replay exported validation without QA writes."""
import json
import math
import sys

import numpy as np
import torch

from eval import hand_state_geometry as geometry,hand_state_geometry_data as data,hand_state_geometry_fit as fit
from eval import phrase_native_phase as phase,phrase_native as native,phrase_budget as budget,phrase_state as codec
from eval import joint_uniform_audio as uniform,joint_pilot as old
from eval.joint_model import JointModel,EMPTY,SNAPSHOTS
from eval.joint_phrase import ROOT,_sha,_atomic
from eval.joint_export import read_chart,assert_same


def replay_record(attempt,template,reference,path,record,scales):
    for key in ('ok','reason','attempt_seed','actions'):assert attempt[key]==record[key],key
    assert record['signature_ok']==(phase.base.typed.signature(attempt['source'])==phase.base.typed.signature(template))
    assert record['phase_signature_ok']==(geometry.signature(attempt['source'])==geometry.signature(template))
    assert record['direction_counters_ok']==(geometry.direction_counters(attempt['source'])==geometry.direction_counters(template))
    if not record['ok']:return 0
    actual=read_chart(path/'ExpertPlus.dat',path/'Info.dat');assert_same(attempt['source'],actual,18,0)
    actual['duration_beats']=template['duration_beats']
    data.verify_record(template,actual,record,{'profile':old.profile(template,reference['audio'])})
    profile=old.profile(actual,reference['audio']);ref=old.profile(reference['source'],reference['audio'])
    assert profile==record['profile'] and ref==record['reference_profile']
    assert old.errors(ref,profile,scales)==record['errors']
    return len(actual['notes'])


def audit_data():
    torch.set_num_threads(2);frozen=data.freeze();prepared=json.loads((data.DATA/'prepared.json').read_text())
    before=_sha(data.DATA/'prepared.json');assert prepared['identity']==frozen['identity']
    packet_run=json.loads((codec.OUT/'run.json').read_text());packet_report=json.loads((codec.OUT/'report.json').read_text())
    assert packet_report['identity']==packet_run['identity']
    planner=torch.load(budget.OUT/'model.pt',weights_only=False);bank=torch.load(uniform.OUT/'retrieval.pt',weights_only=False)
    control=JointModel().eval();control.load_state_dict(torch.load(ROOT/frozen['config']['control_checkpoint'],map_location='cpu',weights_only=True))
    events=0;teachers=0;controls=[];literal=0;templates=0
    for family,item in frozen['config']['sources'].items():
        key=family.replace(':','_');reference=torch.load(ROOT/item['payload'],weights_only=False);audio=torch.load(ROOT/item['view'],weights_only=False)
        if item['role']=='train':
            path=data.DATA/'train'/(key+'.pt');assert _sha(path)==prepared['receipts'][family]['sha256']
            actual=torch.load(path,weights_only=True);rebuilt=geometry.teacher(reference['source'],audio)
            assert actual.keys()==rebuilt.keys()
            for name in actual:assert torch.equal(actual[name],rebuilt[name]),(family,name)
            # Independently compare histories with the older lossless state codec.
            packet_path=codec.OUT/'packets'/(key+'.json')
            assert packet_run['config']['source_inventory'][family]['payload_sha256']==item['payload_sha256']
            assert _sha(packet_path)==packet_report['receipts'][family]['packet_sha256']
            steps=[s for s in json.loads(packet_path.read_text())['steps'] if s['action']['gap']>0]
            assert len(steps)==len(actual['count'])
            histories=[];ages=[];targets=[]
            for step in steps:
                history=[EMPTY]*6;age=[0.]*4;row=step['action'];targets.append(row['slots'])
                for hand,prior in enumerate(step['state']['hands']):
                    if prior is None:continue
                    age[hand]=math.log1p((row['target_beat']-prior['beat'])*60/reference['source']['bpm'])/4;age[hand+2]=1.
                    for slot,(c,l,d) in enumerate(prior['notes']):history[hand*3+slot]=(c*3+l)*9+d
                histories.append(history);ages.append(age)
            assert torch.equal(actual['history'],torch.tensor(histories)) and torch.equal(actual['slots'],torch.tensor(targets))
            assert torch.equal(actual['age'],torch.tensor(ages,dtype=torch.float32))
            teachers+=1;events+=len(steps)
            if teachers%50==0:print('audited hand-state teachers',teachers,flush=True)
            continue
        root=data.DATA/'validation'/key;tp=root/'template.json';receipt=prepared['receipts'][family]
        assert _sha(tp)==receipt['template_sha256'];saved=json.loads(tp.read_text())
        source={k:reference['source'][k] for k in ('bpm','duration_beats')};source.update(notes=[],events=[],bombs=[],walls=[])
        plans=native.plan(source,audio,planner,frozen['config']['validation_rate']);assert plans==saved['plans']
        for seed in range(saved['selected_seed']+1):
            attempt=native.render(source,audio,bank,plans,frozen['config']['validation_rate'],seed);generated=attempt.pop('source',None)
            assert codec.canonical(attempt)==json.loads((root/'native-attempts'/f'{seed}.json').read_text())
            assert attempt['ok']==(seed==saved['selected_seed'])
        _,_,template=data.load_validation(frozen,family);assert_same(generated,template,18,0);templates+=1
        for seed in range(2):
            path=root/'control'/str(seed);assert _sha(path/'record.json')==receipt['control_sha256'][seed]
            r=old.cached_record(path/'record.json',frozen['identity']);assert r is not None
            replay=phase.refine(control,template,audio,seed,1.)
            literal+=replay_record(replay,template,reference,path,r,frozen['config']['validation_scales']);controls.append(r)
    assert teachers==336 and events==prepared['training_events'] and templates==6
    assert data.aggregate(controls)==prepared['control'] and _sha(data.DATA/'prepared.json')==before
    data.freeze();result={'identity':frozen['identity'],'teachers_rebuilt':teachers,'literal_teacher_events':events,
        'templates_regenerated':templates,'control_exports_regenerated':len(controls),'control_notes_regenerated':literal,
        'independent_codec_histories_and_physical_ages':True,'prepared_sha256':before,
        'state_packet_report_sha256':_sha(codec.OUT/'report.json'),'protected_unchanged':True,'audit_code_sha256':_sha(__file__)}
    _atomic(data.DATA/'artifact-audit.json',result);return result


def audit_validation():
    torch.set_num_threads(2);frozen,parent,prepared=fit.freeze();selection=json.loads((fit.OUT/'selection.json').read_text())
    before=_sha(fit.OUT/'selection.json');trained=json.loads((fit.OUT/'training.json').read_text());summaries=[];records_by_update={};literal=0
    assert selection['identity']==trained['identity']==frozen['identity'] and trained['status']=='FIT_COMPLETE'
    for update in SNAPSHOTS:
        path=fit.OUT/f'snapshot-{update}.pt';sha=_sha(path);assert sha==trained['snapshots'][str(update)]
        model=geometry.GeometryModel().eval();model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True));records=[]
        for family,item in parent['config']['sources'].items():
            if item['role']!='validation':continue
            reference,audio,template=data.load_validation(parent,family)
            for seed in range(2):
                target=fit.OUT/'validation'/str(update)/family.replace(':','_')/str(seed)
                r=old.cached_record(target/'record.json',frozen['identity']+':'+sha);assert r is not None
                replay=geometry.refine(model,template,audio,seed,1.)
                literal+=replay_record(replay,template,reference,target,r,parent['config']['validation_scales']);records.append(r)
        summary={**data.aggregate(records),'update':update,'snapshot_sha256':sha}
        assert summary==json.loads((fit.OUT/'validation'/f'summary-{update}.json').read_text())
        summaries.append(summary);records_by_update[update]=records
        print('audited hand-state validation snapshot',update,flush=True)
    chosen=fit.choose(summaries);gates=fit.qualify(chosen,prepared['control'],records_by_update[chosen['update']])
    assert summaries==selection['snapshots'] and chosen['update']==selection['selected'] and chosen['snapshot_sha256']==selection['snapshot_sha256']
    assert gates==selection['gates'] and selection['status']==('VALIDATION_POSITIVE' if all(gates.values()) else 'VALIDATION_NEGATIVE')
    assert _sha(fit.OUT/'selection.json')==before;fit.freeze()
    result={'identity':frozen['identity'],'validation_exports_regenerated':48,'literal_notes_regenerated':literal,
        'selection_sha256':before,'selected_update':chosen['update'],'protected_unchanged':True,'audit_code_sha256':_sha(__file__)}
    _atomic(fit.OUT/'validation-audit.json',result);return result


if __name__=='__main__':print(json.dumps({'data':audit_data,'validation':audit_validation}[sys.argv[1]](),indent=1))
