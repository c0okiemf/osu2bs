"""Reconstruct approved teacher data and verify all continuous-model output artifacts."""
from collections import Counter
import json

import numpy as np
import torch

from eval import joint_approved_fit as candidate,joint_ordered as ordered,joint_pilot as old
from eval.joint_model import KEYS,SNAPSHOTS,actions,action_notes
from eval.joint_phrase import ROOT,_atomic,_sha,read_source
from eval.joint_export import read_chart
from eval.joint_deployment import reference_view
from eval.joint_continuity import describe


def preparation_audit(frozen,prepared):
    train=set();validation=set();contexts=0;source_count=0
    for family,item in sorted(frozen['config']['sources'].items()):
        original=torch.load(ROOT/item['payload'],weights_only=False)
        current=read_source(item['record'])
        for key in ('notes','bombs','walls'):
            assert list(map(tuple,current[key]))==list(map(tuple,original['source'][key]))
        for key in ('bpm','duration_beats'):assert current[key]==original['source'][key]
        assert action_notes(actions(current))==list(map(tuple,current['notes']));source_count+=1
        if item['role']=='validation':validation.add(family);continue
        train.add(family);expected=candidate.teacher(original,candidate.load_view(family,frozen),'train')
        actual=torch.load(candidate.OUT/'data'/(family.replace(':','_')+'.pt'),weights_only=False)
        assert set(actual)==set(KEYS)
        for key in KEYS:assert torch.equal(expected[key],actual[key]),(family,key)
        contexts+=len(actual['gap'])
    assert len(train)==45 and validation==candidate.VALIDATION and not train&validation
    assert not (train|validation)&set(ordered.inputs()) and 'fam:24227' not in train
    initial=torch.load(candidate.OUT/'snapshot-0.pt',weights_only=True)
    old_initial=torch.load(old.OUT/'snapshot-0.pt',weights_only=True)
    for key in initial:assert torch.equal(initial[key],old_initial[key]),key
    state=torch.load(candidate.OUT/'train_state.pt',map_location='cpu',weights_only=False)
    assert state['identity']==frozen['identity'] and state['update']==6000 and state['bos_crops']>=24000
    last=torch.load(candidate.OUT/'snapshot-6000.pt',map_location='cpu',weights_only=True)
    for key in last:assert torch.equal(last[key],state['model'][key]),key
    return {'verified_sources':source_count,'train_families':len(train),'validation_families':len(validation),
            'exact_teacher_actions_and_contexts':contexts,'fresh_original_initialization':True,'completed_updates':state['update']}


def run():
    torch.set_num_threads(2);frozen,prepared=candidate.prepare();preparation=preparation_audit(frozen,prepared)
    summaries=[];validation_exports=0
    for update in SNAPSHOTS:
        sha=_sha(candidate.OUT/f'snapshot-{update}.pt');identity=frozen['identity']+':'+sha;records=[]
        for family,item in sorted(frozen['config']['sources'].items()):
            if item['role']!='validation':continue
            data=torch.load(ROOT/item['payload'],weights_only=False)
            for seed in range(2):
                root=candidate.OUT/'validation'/str(update)/family.replace(':','_')/str(seed)
                rec=old.cached_record(root/'record.json',identity);assert rec is not None;records.append(rec)
                if rec['artifacts']:
                    actual=read_chart(root/'ExpertPlus.dat',root/'Info.dat');actual['duration_beats']=data['source']['duration_beats']
                    assert actual['bpm']==data['source']['bpm'] and actual['bombs']==actual['walls']==[]
                    assert actual['settings']['njs']==18 and actual['settings']['offset_beats']==0
                    validation_exports+=1
                    if rec['ok']:
                        profile=old.profile(actual,data['audio']);reference=old.profile(data['source'],data['audio'])
                        assert profile==rec['profile'] and reference==rec['reference_profile']
                        assert old.errors(reference,profile,prepared['validation_scales'])==rec['errors']
                elif rec['ok']:raise AssertionError('complete validation attempt has no export')
        assert len(records)==12
        summary=candidate.validation_summary(update,sha,records)
        assert summary==json.loads((candidate.OUT/'validation'/f'summary-{update}.json').read_text());summaries.append(summary)
    selection=json.loads((candidate.OUT/'selection.json').read_text());chosen=candidate.choose_snapshot(summaries)
    assert selection['snapshots']==summaries and selection['selected']==chosen['update']
    assert selection['snapshot_sha256']==chosen['snapshot_sha256'] and selection['identity']==frozen['identity']
    scales=old.evaluation_freeze()['config']['scales'];songs=[];evidence={};verdicts=Counter();exports=0
    identity=frozen['identity']+':'+selection['snapshot_sha256']
    for family,item in ordered.inputs().items():
        key=family.replace(':','_');root=candidate.OUT/'development'/key;song=json.loads((root/'song.json').read_text());songs.append(song);records=[]
        for seed in range(6):
            target=root/'ordered'/str(seed);rec=old.cached_record(target/'record.json',identity);assert rec is not None;records.append(rec)
            verdicts[rec.get('machine',{}).get('verdict',rec['reason'])]+=1
            if not rec['ok']:continue
            actual=read_chart(target/'ExpertPlus.dat',target/'Info.dat');actual['duration_beats']=item['source']['duration_beats']
            assert actual['bombs']==item['source']['bombs'] and actual['walls']==item['source']['walls']
            assert actual['bpm']==item['source']['bpm'] and actual['settings']['njs']==18 and actual['settings']['offset_beats']==0
            view=reference_view(actual,item['data']['source']);profile=old.profile(view,item['data']['audio']);reference=old.profile(item['data']['source'],item['data']['audio'])
            assert profile==rec['profile'] and reference==rec['reference_profile']
            assert old.errors(reference,profile,scales)==rec['errors'];exports+=1
        selected=ordered.select(records,item['b0']);assert selected==song['arms']['ordered'] and song['identity']==identity
        cp=old.PILOT/'development'/key/'b0' if selected['fallback'] else root/'ordered'/str(selected['selected_seed'])
        actual=read_chart(cp/'ExpertPlus.dat',cp/'Info.dat')
        evidence[family]={'selected':describe(actual),'selected_gaps':candidate.gap_diagnostic(actual),
            'attempt_gaps':[candidate.gap_diagnostic(read_chart(root/'ordered'/str(i)/'ExpertPlus.dat',root/'ordered'/str(i)/'Info.dat')) if r['ok'] else None for i,r in enumerate(records)]}
    before=_sha(candidate.OUT/'report.json');candidate.decide(songs,frozen,selection,evidence)
    assert before==_sha(candidate.OUT/'report.json')
    result={**preparation,'validation_exports':validation_exports,'development_exports':exports,'attempts':dict(verdicts),
            'repeat_report_sha256':before,'audit_code_sha256':_sha(__file__)}
    _atomic(candidate.OUT/'artifact-audit.json',result);return result


if __name__=='__main__':print(json.dumps(run(),indent=1))
