"""Reconstruct both matched approved banks and all actual comparison exports."""
from collections import Counter
import json

import numpy as np
import torch

from eval import joint_expanded as experiment,joint_quarantine as prior,joint_compatible as compatible,joint_ordered as ordered,joint_pilot as old
from eval.joint_continuity import describe
from eval.joint_deployment import reference_view
from eval.joint_export import read_chart
from eval.joint_phrase import ROOT,_atomic,_sha


def run():
    frozen,receipt=experiment.prepare();expected=experiment.make_banks(*experiment.bank_inputs())
    banks={a:torch.load(experiment.OUT/(a+'.pt'),weights_only=False) for a in expected}
    for arm,bank in banks.items():np.testing.assert_equal(bank,expected[arm])
    n=len(banks['approved24']['entries'])
    np.testing.assert_array_equal(banks['approved24']['x'],banks['ordered']['x'][:n])
    for key in ('mean','sd'):np.testing.assert_array_equal(banks['approved24'][key],banks['ordered'][key])
    report_path=experiment.OUT/'report.json';before=_sha(report_path);scales=old.evaluation_freeze()['config']['scales']
    songs=[];controls={};evidence={};counts={a:Counter() for a in banks};verified=Counter();reused=Counter()
    for family,item in ordered.inputs().items():
        key=family.replace(':','_');dest=experiment.OUT/'development'/key;song=json.loads((dest/'song.json').read_text());songs.append(song)
        previous=json.loads((prior.OUT/'development'/key/'song.json').read_text());all_audits={};sources={}
        for arm,bank in banks.items():
            records=[];audits=[]
            for seed in range(6):
                target=dest/arm/str(seed);r=old.cached_record(target/'record.json',frozen['identity']);assert r is not None
                records.append(r);counts[arm][r.get('machine',{}).get('verdict',r.get('reason'))]+=1;audit=None
                if r['ok']:
                    actual=read_chart(target/'ExpertPlus.dat',target/'Info.dat');actual['duration_beats']=item['source']['duration_beats']
                    assert actual['bombs']==item['source']['bombs'] and actual['walls']==item['source']['walls']
                    assert actual['bpm']==item['source']['bpm'] and actual['settings']['njs']==18 and actual['settings']['offset_beats']==0
                    audit=compatible.audit_ledger(actual,r['donors'],bank)
                    assert audit['compressed']==audit['unknown']==audit['nonterminal_unknown_exits']==0
                    view=reference_view(actual,item['data']['source']);profile=old.profile(view,item['data']['audio']);reference=old.profile(item['data']['source'],item['data']['audio'])
                    assert profile==r['profile'] and reference==r['reference_profile'] and old.errors(reference,profile,scales)==r['errors'];verified[arm]+=1
                    if 'qa_reused_from' in r:
                        p=ROOT/r['qa_reused_from'];assert _sha(p)==r['qa_reused_record_sha256'];saved=json.loads(p.read_text())
                        assert saved['artifacts']==r['artifacts'] and saved['machine']==r['machine'];reused[arm]+=1
                audits.append(audit)
            chosen=ordered.select(records,item['b0']);assert chosen==song['arms'][arm];all_audits[arm]=audits
            p=old.PILOT/'development'/key/'b0' if chosen['fallback'] else dest/arm/str(chosen['selected_seed'])
            sources[arm]=read_chart(p/'ExpertPlus.dat',p/'Info.dat');sources[arm]['duration_beats']=item['source']['duration_beats']
        controls[family]={**previous,'arms':{**previous['arms'],'ordered':song['arms']['approved24']}}
        a=song['arms']['approved24']
        evidence[family]={'attempts':all_audits['ordered'],'selected_seed':song['arms']['ordered']['selected_seed'],
            'approved24_attempts':all_audits['approved24'],
            'matched_approved24_control_audit':None if a['fallback'] else compatible.audit_ledger(sources['approved24'],a['selected']['donors'],banks['approved24']),
            'continuity_control':describe(sources['approved24']),'continuity_candidate':describe(sources['ordered'])}
    experiment.decide(songs,frozen,controls,evidence);assert before==_sha(report_path)
    result={'verified_complete_exports':dict(verified),'attempts':{a:dict(v) for a,v in counts.items()},'qa_reused':dict(reused),
            'both_banks_rebuilt_exact':True,'old_rows_and_moments_exact':True,'repeat_report_sha256':before,'audit_code_sha256':_sha(__file__)}
    _atomic(experiment.OUT/'artifact-audit.json',result);return result


if __name__=='__main__':print(json.dumps(run(),indent=1))
