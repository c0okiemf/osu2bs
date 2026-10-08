"""Regenerate phase refinements and verify native directional roles without QA writes."""
from collections import Counter
import json

import torch

from eval import phrase_native_phase as candidate,joint_refine as typed,joint_uniform_audio as uniform,joint_pilot as old
from eval.joint_model import JointModel
from eval.joint_phrase import ROOT,_sha,_atomic,encode_events
from parity import family as cut_family
from eval.joint_deployment import reference_view
from eval.joint_export import read_chart
from qa.comparator_v2 import chart_verdict


def run():
    torch.set_num_threads(2);frozen=candidate.freeze();before=_sha(candidate.OUT/'report.json')
    model=JointModel().eval();model.load_state_dict(torch.load(ROOT/frozen['config']['checkpoint'],map_location='cpu',weights_only=True))
    scales=old.evaluation_freeze()['config']['scales'];songs=[];proofs={};verdicts=Counter();verified=0;reused=0;notes=0
    for family,item in candidate.inputs().items():
        key=family.replace(':','_');root=candidate.OUT/'development'/key;records=[]
        audio=torch.load(uniform.OUT/'audio'/key/'features.pt',weights_only=False)
        for seed,temperature in enumerate(candidate.TEMPERATURES):
            path=root/'ordered'/str(seed);r=old.cached_record(path/'record.json',frozen['identity']);assert r is not None
            replay=candidate.refine(model,item['template'],audio,seed,temperature)
            for name in ('ok','reason','attempt_seed','actions'):assert r[name]==replay[name],(family,seed,name)
            assert r['signature_ok']==(typed.signature(replay['source'])==typed.signature(item['template']))
            assert r['phase_signature_ok']==(candidate.signature(replay['source'])==candidate.signature(item['template']))
            assert r['direction_counters_ok']==(candidate.direction_counters(replay['source'])==candidate.direction_counters(item['template']))
            assert r['template_artifacts']==item['template_record']['artifacts']
            records.append(r);verdicts[r.get('machine',{}).get('verdict',r['reason'])]+=1
            if not r['ok']:continue
            actual=read_chart(path/'ExpertPlus.dat',path/'Info.dat');actual['duration_beats']=item['template']['duration_beats']
            for name in ('notes','bombs','walls'):assert list(map(tuple,actual[name]))==list(map(tuple,replay['source'][name])),(family,seed,name)
            candidate.verify_export(item['template'],actual,r,item['template_record'])
            template_events=encode_events(item['template']['notes']);actual_events=encode_events(actual['notes'])
            assert len(template_events)==len(actual_events)
            for template_event,actual_event in zip(template_events,actual_events):
                assert template_event['beat']==actual_event['beat']
                for old_hand,new_hand in zip(template_event['hands'],actual_event['hands']):
                    assert len(old_hand)==len(new_hand)
                    for old_note,new_note in zip(old_hand,new_hand):
                        a,b=old_note[2],new_note[2]
                        assert a==b if a in (2,3,8) else b not in (2,3,8) and cut_family(a)==cut_family(b)
            view=reference_view(actual,item['data']['source']);profile=old.profile(view,item['data']['audio'])
            reference=old.profile(item['data']['source'],item['data']['audio'])
            assert profile==r['profile'] and reference==r['reference_profile'] and old.errors(reference,profile,scales)==r['errors']
            assert r['machine']['verdict']==chart_verdict(r['machine']['contradictions'],r['machine']['support'])
            assert r['machine']['admitted']==(r['machine']['verdict']=='MACHINE_PASS_QUALITY_NOT_EVALUATED')
            if 'qa_reused_from' in r:
                prior=ROOT/r['qa_reused_from'];assert _sha(prior)==r['qa_reused_record_sha256'];saved=json.loads(prior.read_text())
                assert saved['artifacts']==r['artifacts'] and saved['machine']==r['machine'];reused+=1
            verified+=1;notes+=len(actual['notes'])
        chosen=candidate.select(records,item['template_record'],item['template_seed'])
        song=json.loads((root/'song.json').read_text());assert song['arms']['ordered']==chosen
        songs.append(song);proofs[family]=candidate.evidence(item,records,chosen,root)
        print('verified native phase refinement',family,flush=True)
    candidate.decide(songs,frozen,proofs);assert _sha(candidate.OUT/'report.json')==before;candidate.freeze()
    result={'identity':frozen['identity'],'verified_complete_exports':verified,'literal_notes_regenerated':notes,
        'attempts':dict(verdicts),'qa_reused':reused,'repeat_report_sha256':before,'protected_production_unchanged':True,
        'exact_timing_types_environments_rhythm_and_support_masks':True,'exact_phase_lateral_cuts_and_directional_categories':True,'audit_code_sha256':_sha(__file__)}
    _atomic(candidate.OUT/'artifact-audit.json',result);return result


if __name__=='__main__':print(json.dumps(run(),indent=1))
