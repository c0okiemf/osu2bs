"""Reselect only the four existing checkpoints using their actual serving decoder."""
import json
import time
from unittest.mock import patch

import torch

from eval import joint_censored_timing as control,joint_approved_fit as fit,joint_pilot as old,joint_ordered as ordered
from eval.expressive_manifest import freeze_run
from eval.joint_model import SNAPSHOTS
from eval.joint_phrase import ROOT,_atomic,_sha
from eval.joint_export import read_chart
from eval.joint_deployment import reference_view
from eval.joint_continuity import describe

OUT=ROOT/'experiments/joint-phrase-v18/aligned-checkpoint'
SPEC=ROOT/'docs/specs/2026-09-30-aligned-checkpoint-design.md'


def freeze_experiment():
    parent,fitrun,old_selection=control.freeze_experiment();report=json.loads((control.OUT/'report.json').read_text())
    if report['identity']!=parent['identity'] or not report['gates']['complete_panel']:raise ValueError('corrected decoder control incomplete')
    return freeze_run(OUT,{'parent_identity':parent['identity'],'code_sha256':_sha(__file__),'spec_sha256':_sha(SPEC),
        'decoder_sha256':_sha(control.__file__),'checkpoints':{str(s):_sha(fit.OUT/f'snapshot-{s}.pt') for s in SNAPSHOTS},
        'control_report_sha256':_sha(control.OUT/'report.json'),'control_audit_sha256':_sha(control.OUT/'artifact-audit.json'),
        'control_artifacts':{str(p.relative_to(ROOT)):_sha(p) for root in (control.OUT/'validation',control.OUT/'development') for p in root.rglob('*')
            if p.is_file() and p.name in ('record.json','song.json','ExpertPlus.dat','Info.dat')}}),fitrun


def validation_path(update,family,seed):
    return (control.OUT/'validation' if update==6000 else OUT/'validation'/str(update))/family.replace(':','_')/str(seed)


def validate(deadline_seconds=2700):
    deadline=time.monotonic()+min(deadline_seconds,2700);torch.set_num_threads(2);frozen,parent=freeze_experiment()
    prepared=json.loads((fit.OUT/'prepared.json').read_text());summaries=[]
    for update in SNAPSHOTS:
        sha=frozen['config']['checkpoints'][str(update)];model=control.load_model({'selected':update});records=[]
        for family in sorted(fit.VALIDATION):
            data=torch.load(ROOT/parent['config']['sources'][family]['payload'],weights_only=False);audio=fit.load_view(family,parent)
            for seed in range(2):
                target=validation_path(update,family,seed)
                identity=frozen['config']['parent_identity'] if update==6000 else frozen['identity']+':'+sha
                record=old.cached_record(target/'record.json',identity)
                if record is None:
                    if update==6000:raise ValueError('missing authenticated control attempt')
                    if time.monotonic()>=deadline:return {'status':'INCOMPLETE','update':update,'family':family,'seed':seed}
                    attempt=control.rollout(model,old.generation_source(data),audio,prepared['validation_rate'],seed=seed)
                    record=old.measure_attempt(attempt,data,target,identity,prepared['validation_scales'])
                    print('aligned validation',update,family,seed,record['ok'],record['reason'],flush=True)
                records.append(record)
        summary=fit.validation_summary(update,sha,records);_atomic(OUT/'validation'/f'summary-{update}.json',summary);summaries.append(summary)
    chosen=fit.choose_snapshot(summaries)
    result={'identity':frozen['identity'],'status':'VALIDATION_COMPLETE','snapshots':summaries,'selected':chosen['update'],
        'snapshot_sha256':chosen['snapshot_sha256'],'learned_candidate':chosen['update']>0,
        'selection':'failures; mean rhythm+geometry; earliest update','new_validation_attempts':36,'reused_validation_attempts':12}
    _atomic(OUT/'selection.json',result);return result


def finish(frozen,selection):
    if selection['selected']==6000:
        result={**json.loads((control.OUT/'report.json').read_text()),'identity':frozen['identity'],'selection':selection,
            'inherited_result':{'root':str(control.OUT.relative_to(ROOT)),'report_sha256':_sha(control.OUT/'report.json'),
                'audit_sha256':_sha(control.OUT/'artifact-audit.json'),'new_development_attempts':0},
            'budget':'36 new validation attempts;12 validation and48 development attempts reused; no fit or new development generation'}
    elif selection['selected']==0:
        result={'identity':frozen['identity'],'status':'NO_LEARNED_CANDIDATE','release_eligible':False,'selection':selection,
                'new_development_attempts':0,'reason':'unchanged validation rule selected the initialization checkpoint'}
    else:
        families=[r['family'] for r in json.loads((control.OUT/'report.json').read_text())['families']]
        songs=[json.loads((OUT/'development'/f.replace(':','_')/'song.json').read_text()) for f in families]
        with patch.object(ordered,'OUT',OUT):result=ordered.decide(songs,frozen)
        evidence={}
        for song in songs:
            family=song['family'];key=family.replace(':','_');arm=song['arms']['ordered'];root=OUT/'development'/key
            cp=old.PILOT/'development'/key/'b0' if arm['fallback'] else root/'ordered'/str(arm['selected_seed'])
            actual=read_chart(cp/'ExpertPlus.dat',cp/'Info.dat')
            evidence[family]={'selected':describe(actual),'selected_gaps':fit.gap_diagnostic(actual),
                'attempt_gaps':[fit.gap_diagnostic(read_chart(root/'ordered'/str(i)/'ExpertPlus.dat',root/'ordered'/str(i)/'Info.dat')) if r['ok'] else None for i,r in enumerate(arm['attempts'])]}
        result.update(selection=selection,continuity_and_gaps=evidence,
            arm_semantics={'ordered':'approved-only checkpoint chosen with unchanged rule and actual censored decoder','retrieval':'original deployment retrieval','b0':'production'},
            budget='36 new validation attempts;12 reused;48 development attempts; no fit',
            claim='opened development; original rule with aligned decoder, not confirmation or universal quality')
    _atomic(OUT/'report.json',result);return result


def development(deadline_seconds=2700):
    deadline=time.monotonic()+min(deadline_seconds,2700);torch.set_num_threads(2);frozen,parent=freeze_experiment()
    selection=json.loads((OUT/'selection.json').read_text())
    if selection['identity']!=frozen['identity'] or selection['snapshot_sha256']!=frozen['config']['checkpoints'][str(selection['selected'])]:raise ValueError('aligned selection changed')
    if selection['selected'] in (0,6000):return finish(frozen,selection)
    model=control.load_model(selection);kit=old.machine_tools();scales=old.evaluation_freeze()['config']['scales']
    for family,item in ordered.inputs().items():
        key=family.replace(':','_');root=OUT/'development'/key;priorroot=control.OUT/'development'/key
        prior=json.loads((priorroot/'song.json').read_text());audio=fit.load_view(family,parent);records=[]
        for seed,temp in enumerate((.85,1.,1.15,.85,1.,1.15)):
            target=root/'ordered'/str(seed);rec=old.cached_record(target/'record.json',frozen['identity'])
            if rec is None:
                if time.monotonic()>=deadline:return {'status':'INCOMPLETE','family':family,'seed':seed}
                attempt=control.rollout(model,item['source'],audio,item['rate'],seed=seed,temperature=temp)
                rec=ordered.measure(attempt,item['data'],target,frozen['identity'],scales,kit,priorroot/'ordered'/str(seed)/'record.json')
                print('aligned development',family,seed,rec['ok'],rec.get('machine',{}).get('verdict',rec['reason']),flush=True)
            if rec['ok']:
                actual=read_chart(target/'ExpertPlus.dat',target/'Info.dat');actual['duration_beats']=item['source']['duration_beats'];control.audit_targets(rec,actual)
            records.append(rec)
        chosen=ordered.select(records,item['b0']);song={**prior,'identity':frozen['identity'],'arms':{'retrieval':prior['arms']['retrieval'],'ordered':chosen}}
        _atomic(root/'song.json',song)
    freeze_experiment();return finish(frozen,selection)


def audit():
    torch.set_num_threads(2);parent,prepared=fit.prepare();frozen,parent=freeze_experiment();summaries=[];validated=0
    for update in SNAPSHOTS:
        sha=frozen['config']['checkpoints'][str(update)];records=[]
        for family in sorted(fit.VALIDATION):
            data=torch.load(ROOT/parent['config']['sources'][family]['payload'],weights_only=False)
            for seed in range(2):
                root=validation_path(update,family,seed);identity=frozen['config']['parent_identity'] if update==6000 else frozen['identity']+':'+sha
                rec=old.cached_record(root/'record.json',identity);assert rec is not None;records.append(rec)
                if not rec['ok']:continue
                actual=read_chart(root/'ExpertPlus.dat',root/'Info.dat');actual['duration_beats']=data['source']['duration_beats']
                assert actual['bpm']==data['source']['bpm'] and actual['bombs']==actual['walls']==[]
                assert actual['settings']['njs']==18 and actual['settings']['offset_beats']==0
                control.audit_targets(rec,actual);profile=old.profile(actual,data['audio']);reference=old.profile(data['source'],data['audio'])
                assert profile==rec['profile'] and reference==rec['reference_profile']
                assert old.errors(reference,profile,prepared['validation_scales'])==rec['errors'];validated+=1
        summary=fit.validation_summary(update,sha,records);assert summary==json.loads((OUT/'validation'/f'summary-{update}.json').read_text());summaries.append(summary)
    selection=json.loads((OUT/'selection.json').read_text());chosen=fit.choose_snapshot(summaries)
    assert selection['identity']==frozen['identity'] and selection['snapshots']==summaries and selection['selected']==chosen['update'] and selection['snapshot_sha256']==chosen['snapshot_sha256']
    exports=0;scales=old.evaluation_freeze()['config']['scales']
    if selection['selected'] not in (0,6000):
        for family,item in ordered.inputs().items():
            root=OUT/'development'/family.replace(':','_');song=json.loads((root/'song.json').read_text());records=[]
            for seed in range(6):
                target=root/'ordered'/str(seed);rec=old.cached_record(target/'record.json',frozen['identity']);assert rec is not None;records.append(rec)
                if not rec['ok']:continue
                actual=read_chart(target/'ExpertPlus.dat',target/'Info.dat');actual['duration_beats']=item['source']['duration_beats']
                assert actual['bpm']==item['source']['bpm'] and actual['bombs']==item['source']['bombs'] and actual['walls']==item['source']['walls']
                assert actual['settings']['njs']==18 and actual['settings']['offset_beats']==0
                control.audit_targets(rec,actual);view=reference_view(actual,item['data']['source'])
                profile=old.profile(view,item['data']['audio']);reference=old.profile(item['data']['source'],item['data']['audio'])
                assert profile==rec['profile'] and reference==rec['reference_profile']
                assert old.errors(reference,profile,scales)==rec['errors'];exports+=1
                if 'qa_reused_from' in rec:
                    path=ROOT/rec['qa_reused_from'];assert _sha(path)==rec['qa_reused_record_sha256']
                    prior=json.loads(path.read_text());assert prior['artifacts']==rec['artifacts'] and prior['machine']==rec['machine']
            assert song['arms']['ordered']==ordered.select(records,item['b0'])
    before=_sha(OUT/'report.json');finish(frozen,selection);assert before==_sha(OUT/'report.json')
    result={'verified_validation_exports':validated,'new_development_exports':exports,'selected':selection['selected'],
        'inherited_development':selection['selected']==6000,'all_parent_artifacts_verified':True,'repeat_report_sha256':before}
    _atomic(OUT/'artifact-audit.json',result);return result


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=['validate','development','audit']);args=parser.parse_args()
    print(json.dumps({'validate':validate,'development':development,'audit':audit}[args.command](),indent=1))
