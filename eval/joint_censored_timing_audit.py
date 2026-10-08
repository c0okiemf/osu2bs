"""Verify raw-target preservation and all exported censored-timing comparisons."""
from collections import Counter
import json

import numpy as np
import torch

from eval import joint_censored_timing as candidate,joint_approved_fit as fit,joint_pilot as old,joint_ordered as ordered
from eval.joint_phrase import ROOT,_atomic,_sha
from eval.joint_export import read_chart
from eval.joint_deployment import reference_view
from eval.joint_continuity import describe


def run():
    torch.set_num_threads(2);parent,prepared=fit.prepare();frozen,parent,selection=candidate.freeze_experiment()
    validation=json.loads((candidate.OUT/'validation-report.json').read_text());probe=json.loads((candidate.OUT/'raw-boundary-probe.json').read_text())
    assert validation['raw_probe_sha256']==_sha(candidate.OUT/'raw-boundary-probe.json') and probe['identity']==frozen['identity']
    rows=probe['rows'];assert len(rows)==probe['tiny_decisions'] and probe['exact_replayed_exports']==12
    assert sum(r['raw_mode']==r['category'] for r in rows)==probe['raw_modal']
    assert sum(r['rank']==1 for r in rows)==probe['legal_modal']
    assert np.quantile([r['raw_probability'] for r in rows],[0,.1,.5,.9,1]).tolist()==probe['raw_probability_quantiles']
    validation_exports=0;validation_rows=[]
    for family in sorted(fit.VALIDATION):
        data=torch.load(ROOT/parent['config']['sources'][family]['payload'],weights_only=False)
        for seed in range(2):
            dest=candidate.OUT/'validation'/family.replace(':','_')/str(seed);rec=old.cached_record(dest/'record.json',frozen['identity']);assert rec is not None
            priorroot=fit.OUT/'validation'/str(selection['selected'])/family.replace(':','_')/str(seed);prior=json.loads((priorroot/'record.json').read_text());gaps=None;audit=None
            if rec['ok']:
                actual=read_chart(dest/'ExpertPlus.dat',dest/'Info.dat');actual['duration_beats']=data['source']['duration_beats']
                assert actual['bpm']==data['source']['bpm'] and actual['bombs']==actual['walls']==[]
                assert actual['settings']['njs']==18 and actual['settings']['offset_beats']==0
                audit=candidate.audit_targets(rec,actual);gaps=fit.gap_diagnostic(actual);validation_exports+=1
                profile=old.profile(actual,data['audio']);reference=old.profile(data['source'],data['audio'])
                assert profile==rec['profile'] and reference==rec['reference_profile']
                assert old.errors(reference,profile,prepared['validation_scales'])==rec['errors']
            validation_rows.append({'family':family,'seed':seed,'ok':rec['ok'],'reason':rec['reason'],'errors':rec.get('errors'),
                'control_errors':prior.get('errors'),'gaps':gaps,'control_gaps':fit.gap_diagnostic(read_chart(priorroot/'ExpertPlus.dat',priorroot/'Info.dat')),'target_audit':audit})
    assert validation_rows==validation['rows'] and validation_exports==validation['complete']
    songs=[];evidence={};verdicts=Counter();exports=0;reused=0;scales=old.evaluation_freeze()['config']['scales']
    for family,item in ordered.inputs().items():
        key=family.replace(':','_');root=candidate.OUT/'development'/key;song=json.loads((root/'song.json').read_text());songs.append(song);records=[];audits=[]
        for seed in range(6):
            target=root/'ordered'/str(seed);rec=old.cached_record(target/'record.json',frozen['identity']);assert rec is not None;records.append(rec);audit=None
            verdicts[rec.get('machine',{}).get('verdict',rec['reason'])]+=1
            if rec['ok']:
                actual=read_chart(target/'ExpertPlus.dat',target/'Info.dat');actual['duration_beats']=item['source']['duration_beats']
                assert actual['bpm']==item['source']['bpm'] and actual['bombs']==item['source']['bombs'] and actual['walls']==item['source']['walls']
                assert actual['settings']['njs']==18 and actual['settings']['offset_beats']==0
                audit=candidate.audit_targets(rec,actual);view=reference_view(actual,item['data']['source'])
                profile=old.profile(view,item['data']['audio']);reference=old.profile(item['data']['source'],item['data']['audio'])
                assert profile==rec['profile'] and reference==rec['reference_profile']
                assert old.errors(reference,profile,scales)==rec['errors'];exports+=1
                if 'qa_reused_from' in rec:
                    path=ROOT/rec['qa_reused_from'];assert _sha(path)==rec['qa_reused_record_sha256']
                    prior=json.loads(path.read_text());assert prior['artifacts']==rec['artifacts'] and prior['machine']==rec['machine'];reused+=1
            audits.append(audit)
        selected=ordered.select(records,item['b0']);assert song['arms']['ordered']==selected and song['identity']==frozen['identity']
        cp=old.PILOT/'development'/key/'b0' if selected['fallback'] else root/'ordered'/str(selected['selected_seed'])
        actual=read_chart(cp/'ExpertPlus.dat',cp/'Info.dat')
        evidence[family]={'selected':describe(actual),'selected_gaps':fit.gap_diagnostic(actual),'target_audits':audits,
            'attempt_gaps':[fit.gap_diagnostic(read_chart(root/'ordered'/str(i)/'ExpertPlus.dat',root/'ordered'/str(i)/'Info.dat')) if r['ok'] else None for i,r in enumerate(records)]}
    before=_sha(candidate.OUT/'report.json');candidate.decide(songs,frozen,selection,evidence);assert before==_sha(candidate.OUT/'report.json')
    result={'validation_exports':validation_exports,'development_exports':exports,'attempts':dict(verdicts),'qa_reused':reused,
            'all_event_targets_unmodified':True,'raw_probe_verified':True,'repeat_report_sha256':before,'audit_code_sha256':_sha(__file__)}
    _atomic(candidate.OUT/'artifact-audit.json',result);return result


if __name__=='__main__':print(json.dumps(run(),indent=1))
