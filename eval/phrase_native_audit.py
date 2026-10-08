"""Replay learned-budget maps and verify exports without writing the QA cache."""
from collections import Counter
import json

import numpy as np
import torch

from eval import phrase_native as candidate,phrase_budget as planner,phrase_state as codec,joint_uniform_audio as control
from eval import joint_ordered as ordered,joint_pilot as old
from eval.joint_deployment import reference_view
from eval.joint_export import read_chart
from eval.joint_phrase import ROOT,_sha,_atomic,phrase_windows
from qa.comparator_v2 import chart_verdict


def run():
    torch.set_num_threads(2);frozen=candidate.freeze();before=_sha(candidate.OUT/'report.json')
    bank=torch.load(control.OUT/'retrieval.pt',weights_only=False);model=torch.load(planner.OUT/'model.pt',weights_only=False)
    training=torch.load(planner.OUT/'training.pt',weights_only=False)
    np.testing.assert_equal(model,planner.fit_ridge(training['x'],training['y'],training['weights'],training['roles']))
    scales=old.evaluation_freeze()['config']['scales'];songs=[];proofs={};verdicts=Counter();verified=0;reused=0
    for family,item in ordered.inputs().items():
        key=family.replace(':','_');dest=candidate.OUT/'development'/key
        audio=torch.load(control.OUT/'audio'/key/'features.pt',weights_only=False)
        plans=candidate.plan(item['source'],audio,model,item['rate']);records=[]
        for seed in range(6):
            target=dest/'ordered'/str(seed);r=old.cached_record(target/'record.json',frozen['identity']);assert r is not None
            replay=candidate.render(item['source'],audio,bank,plans,item['rate'],seed);source=replay.pop('source',None)
            for name,value in replay.items():assert codec.canonical(value)==r[name],(family,seed,name)
            records.append(r);verdicts[r.get('machine',{}).get('verdict',r['reason'])]+=1
            if not r['ok']:continue
            actual=read_chart(target/'ExpertPlus.dat',target/'Info.dat');actual['duration_beats']=item['source']['duration_beats']
            for name in ('notes','bombs','walls'):assert list(map(tuple,actual[name]))==list(map(tuple,source[name])),(family,seed,name)
            assert actual['bpm']==item['source']['bpm'] and actual['settings']['njs']==18 and actual['settings']['offset_beats']==0
            windows=phrase_windows(actual['notes'],actual['bombs'],actual['walls'],actual['duration_beats'])
            last=[None,None]
            for w,entry in zip(windows,r['budget_ledger']):
                assert codec.canonical(last)==entry['prior_hands']
                realized=[0,0,0]
                for event in w['events']:
                    groups=event['hands'];realized[2 if all(groups) else 0 if groups[0] else 1]+=1
                    for hand,notes in enumerate(groups):
                        if notes:last[hand]={'beat':event['beat'],'notes':notes}
                assert realized==entry['realized_counts']
                assert sum(abs(a-b) for a,b in zip(realized,entry['counts']))==entry['absolute_count_error']
            assert len(windows)==len(r['budget_ledger'])==len(plans)
            view=reference_view(actual,item['data']['source']);profile=old.profile(view,item['data']['audio'])
            reference=old.profile(item['data']['source'],item['data']['audio'])
            assert profile==r['profile'] and reference==r['reference_profile'] and old.errors(reference,profile,scales)==r['errors']
            assert r['machine']['verdict']==chart_verdict(r['machine']['contradictions'],r['machine']['support'])
            assert r['machine']['admitted']==(r['machine']['verdict']=='MACHINE_PASS_QUALITY_NOT_EVALUATED')
            if 'qa_reused_from' in r:
                path=ROOT/r['qa_reused_from'];assert _sha(path)==r['qa_reused_record_sha256'];saved=json.loads(path.read_text())
                assert saved['artifacts']==r['artifacts'] and saved['machine']==r['machine'];reused+=1
            verified+=1
        song=json.loads((dest/'song.json').read_text());chosen=ordered.select(records,item['b0'])
        assert chosen==song['arms']['ordered'];songs.append(song);proofs[family]=candidate.evidence(item,records,chosen,bank,dest)
        print('verified native budget',family,flush=True)
    candidate.decide(songs,frozen,proofs);assert _sha(candidate.OUT/'report.json')==before
    candidate.freeze()
    result={'identity':frozen['identity'],'verified_complete_exports':verified,'attempts':dict(verdicts),'qa_reused':reused,
        'replayed_all_attempts':True,'rebuilt_ridge_model':True,'repeat_report_sha256':before,'protected_production_unchanged':True,
        'audit_code_sha256':_sha(__file__)}
    _atomic(candidate.OUT/'artifact-audit.json',result);return result


if __name__=='__main__':print(json.dumps(run(),indent=1))
