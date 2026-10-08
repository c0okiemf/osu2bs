"""Re-read quarantined-bank exports and reproduce the frozen report without QA writes."""
from collections import Counter
import json

import numpy as np
import torch

from eval import joint_quarantine as candidate,joint_compatible as control,joint_joins as native,joint_ordered as ordered,joint_pilot as old
from eval.joint_continuity import describe
from eval.joint_deployment import reference_view
from eval.joint_export import read_chart
from eval.joint_phrase import ROOT,_atomic,_sha


def run():
    frozen,receipt=candidate.prepare();bank=torch.load(candidate.OUT/'retrieval.pt',weights_only=False)
    full=torch.load(native.OUT/'retrieval.pt',weights_only=False)
    retained=[e for e in full['entries'] if e['family'] not in frozen['config']['excluded_train_families']]
    raw=np.stack([e['descriptor'] for e in retained]);mean=raw.mean(0);sd=np.maximum(raw.std(0),1e-6)
    expected={'entries':retained,'mean':mean,'sd':sd,'x':(raw-mean)/sd}
    assert set(bank)==set(expected) and len({e['family'] for e in retained})==315
    np.testing.assert_equal(bank['entries'],expected['entries'])
    for key in ('x','mean','sd'):
        np.testing.assert_array_equal(bank[key],expected[key])
    report=json.loads((candidate.OUT/'report.json').read_text());before=_sha(candidate.OUT/'report.json')
    scales=old.evaluation_freeze()['config']['scales'];songs=[];controls={};evidence={};verdicts=Counter();verified=0;reused=0
    for family,item in ordered.inputs().items():
        key=family.replace(':','_');dest=candidate.OUT/'development'/key
        song=json.loads((dest/'song.json').read_text());songs.append(song)
        controls[family]=json.loads((control.OUT/'development'/key/'song.json').read_text())
        records=[];audits=[]
        for seed in range(6):
            target=dest/'ordered'/str(seed);rec=old.cached_record(target/'record.json',frozen['identity'])
            assert rec is not None;records.append(rec);audit=None
            verdicts[rec.get('machine',{}).get('verdict',rec.get('reason'))]+=1
            if rec['ok']:
                actual=read_chart(target/'ExpertPlus.dat',target/'Info.dat')
                actual['duration_beats']=item['source']['duration_beats']
                assert actual['bombs']==item['source']['bombs'] and actual['walls']==item['source']['walls']
                assert actual['bpm']==item['source']['bpm'] and actual['settings']['njs']==18 and actual['settings']['offset_beats']==0
                audit=control.audit_ledger(actual,rec['donors'],bank)
                assert audit['compressed']==audit['unknown']==audit['nonterminal_unknown_exits']==0
                view=reference_view(actual,item['data']['source'])
                profile=old.profile(view,item['data']['audio']);reference=old.profile(item['data']['source'],item['data']['audio'])
                assert profile==rec['profile'] and reference==rec['reference_profile']
                assert old.errors(reference,profile,scales)==rec['errors'];verified+=1
                if 'qa_reused_from' in rec:
                    prior=ROOT/rec['qa_reused_from'];assert _sha(prior)==rec['qa_reused_record_sha256']
                    saved=json.loads(prior.read_text());assert saved['artifacts']==rec['artifacts'] and saved['machine']==rec['machine'];reused+=1
            audits.append(audit)
        assert song['arms']['ordered']==ordered.select(records,item['b0'])
        prior=controls[family]['arms']['ordered'];chosen=song['arms']['ordered']
        pp=old.PILOT/'development'/key/'b0' if prior['fallback'] else control.OUT/'development'/key/'ordered'/str(prior['selected_seed'])
        cp=old.PILOT/'development'/key/'b0' if chosen['fallback'] else dest/'ordered'/str(chosen['selected_seed'])
        previous=read_chart(pp/'ExpertPlus.dat',pp/'Info.dat');previous['duration_beats']=item['source']['duration_beats']
        evidence[family]={'attempts':audits,'selected_seed':chosen['selected_seed'],
            'full_pool_control_audit':None if prior['fallback'] else control.audit_ledger(previous,prior['selected']['donors'],full),
            'continuity_control':describe(previous),'continuity_candidate':describe(read_chart(cp/'ExpertPlus.dat',cp/'Info.dat'))}
    candidate.decide(songs,frozen,controls,evidence)
    assert before==_sha(candidate.OUT/'report.json')
    result={'verified_complete_exports':verified,'attempts':dict(verdicts),'qa_reused':reused,'repeat_report_sha256':before,
            'quarantine_inventory_and_moments_exact':True,'audit_code_sha256':_sha(__file__)}
    _atomic(candidate.OUT/'artifact-audit.json',result);return result


if __name__=='__main__':print(json.dumps(run(),indent=1))
