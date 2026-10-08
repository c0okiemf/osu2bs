"""Rebuild compatible retrieval without confirmed cross-role audio overlap."""
import json
import time
from unittest.mock import patch

import numpy as np
import torch

from eval import joint_compatible as compatible,joint_joins as native,joint_ordered as ordered,joint_pilot as old,joint_audio_identity as audio_audit
from eval.expressive_manifest import freeze_run
from eval.joint_model import _torch_save
from eval.joint_phrase import ROOT,_atomic,_sha
from eval.joint_export import read_chart
from eval.joint_continuity import describe

OUT=ROOT/'experiments/joint-phrase-v12/quarantined-retrieval'
SPEC=ROOT/'docs/specs/2026-09-30-quarantined-retrieval-design.md'


def quarantine_bank(base,excluded):
    if not excluded or not set(excluded)<={e['family'] for e in base['entries']}:raise ValueError('quarantine inventory mismatch')
    entries=[e for e in base['entries'] if e['family'] not in excluded]
    if not entries:raise ValueError('quarantine removed entire bank')
    x=np.stack([e['descriptor'] for e in entries]);mean=x.mean(0);sd=np.maximum(x.std(0),1e-6)
    return {'entries':entries,'mean':mean,'sd':sd,'x':(x-mean)/sd}


def freeze_experiment():
    parent=compatible.freeze_experiment();report=json.loads((compatible.OUT/'report.json').read_text())
    if report['identity']!=parent['identity'] or not report['gates']['complete_panel']:raise ValueError('control incomplete')
    qp=audio_audit.OUT/'quarantine.json';q=json.loads(qp.read_text())
    if q['report_sha256']!=_sha(audio_audit.OUT/'report.json'):raise ValueError('audio quarantine report changed')
    if q['exclude_from_future_fit_roles']!={'train':['fam:24227'],'validation':[]}:raise ValueError('unexpected quarantine scope')
    return freeze_run(OUT,{'parent_identity':parent['identity'],'spec_sha256':_sha(SPEC),'code_sha256':_sha(__file__),
        'control_report_sha256':_sha(compatible.OUT/'report.json'),'quarantine_sha256':_sha(qp),
        'audio_audit_report_sha256':q['report_sha256'],'excluded_train_families':q['exclude_from_future_fit_roles']['train'],
        'control_artifacts':{str(p.relative_to(ROOT)):_sha(p) for p in (compatible.OUT/'development').rglob('*')
            if p.is_file() and p.name in ('song.json','record.json','ExpertPlus.dat','Info.dat')}})


def prepare():
    f=freeze_experiment();path=OUT/'retrieval.pt';rp=OUT/'retrieval-receipt.json'
    if not rp.exists():
        bank=quarantine_bank(torch.load(native.OUT/'retrieval.pt',weights_only=False),f['config']['excluded_train_families'])
        if len({e['family'] for e in bank['entries']})!=315:raise ValueError('retained source inventory changed')
        _torch_save(path,bank);_atomic(rp,{'identity':f['identity'],'sha256':_sha(path),'entries':len(bank['entries']),'families':315})
    receipt=json.loads(rp.read_text())
    if receipt['identity']!=f['identity'] or receipt['sha256']!=_sha(path):raise ValueError('quarantined bank changed')
    return f,receipt


def decide(songs,frozen,controls,evidence):
    with patch.object(compatible,'OUT',OUT):report=compatible.decide(songs,frozen,controls,evidence)
    report['quarantine_ablation']=report.pop('compatibility_search_ablation')
    for d in report['quarantine_ablation'].values():
        d['overlap_exposed_control_mean']=d.pop('native_shortlist_control_mean')
        d['quarantined_bank_mean']=d.pop('compatible_search_mean')
    donors=[d for s in songs for r in s['arms']['ordered']['attempts'] for d in r['donors']]
    report['quarantine_contract']={'excluded':frozen['config']['excluded_train_families'],
        'all_donors_retained':all(d[0] not in frozen['config']['excluded_train_families'] for d in donors),
        'similarity_normalizers':'refitted on315 retained source families only',
        'scope':'donor inventory and its similarity fitting only; historical metric scales, B0 and QA exposure retained; not fresh confirmation'}
    report['arm_semantics']['ordered']='compatible search with confirmed overlapping donor removed and similarity moments refitted'
    report['budget']='six new attempts per family, exact-file QA reuse; historical v10/v7/B0 controls; no model fit'
    _atomic(OUT/'report.json',report);return report


def run(deadline_seconds=2700):
    deadline=time.monotonic()+min(deadline_seconds,2700);torch.set_num_threads(4)
    frozen,receipt=prepare();bank=torch.load(OUT/'retrieval.pt',weights_only=False)
    full=torch.load(native.OUT/'retrieval.pt',weights_only=False)
    scales=old.evaluation_freeze()['config']['scales'];kit=old.machine_tools();songs=[];controls={};evidence={}
    for family,item in ordered.inputs().items():
        key=family.replace(':','_');dest=OUT/'development'/key;root=compatible.OUT/'development'/key
        control=json.loads((root/'song.json').read_text());controls[family]=control;records=[];audits=[]
        for seed in range(6):
            target=dest/'ordered'/str(seed);rec=old.cached_record(target/'record.json',frozen['identity'])
            if rec is None:
                if time.monotonic()>=deadline:return {'status':'INCOMPLETE','family':family,'next_seed':seed,'families':len(songs)}
                started=time.monotonic();attempt=compatible.retrieve(item['source'],item['data']['audio'],bank,item['rate'],seed)
                attempt['elapsed_s']=time.monotonic()-started
                rec=ordered.measure(attempt,item['data'],target,frozen['identity'],scales,kit,root/'ordered'/str(seed)/'record.json')
                print(f'quarantined bank {family} s{seed}: {rec["ok"]} {rec.get("machine",{}).get("verdict")}',flush=True)
            audit=None
            if rec['ok']:
                src=read_chart(target/'ExpertPlus.dat',target/'Info.dat');src['duration_beats']=item['source']['duration_beats']
                audit=compatible.audit_ledger(src,rec['donors'],bank)
                if audit['compressed'] or audit['unknown'] or audit['nonterminal_unknown_exits']:raise ValueError('native context contract failed')
            records.append(rec);audits.append(audit)
        chosen=ordered.select(records,item['b0']);song={**control,'identity':frozen['identity'],
                    'arms':{'retrieval':control['arms']['retrieval'],'ordered':chosen}}
        _atomic(dest/'song.json',song);songs.append(song)
        before=control['arms']['ordered'];prevpath=old.PILOT/'development'/key/'b0' if before['fallback'] else root/'ordered'/str(before['selected_seed'])
        prev=read_chart(prevpath/'ExpertPlus.dat',prevpath/'Info.dat');prev['duration_beats']=item['source']['duration_beats']
        newpath=old.PILOT/'development'/key/'b0' if chosen['fallback'] else dest/'ordered'/str(chosen['selected_seed'])
        evidence[family]={'attempts':audits,'selected_seed':chosen['selected_seed'],
             'full_pool_control_audit':None if before['fallback'] else compatible.audit_ledger(prev,before['selected']['donors'],full),
             'continuity_control':describe(prev),'continuity_candidate':describe(read_chart(newpath/'ExpertPlus.dat',newpath/'Info.dat'))}
    prepare();return decide(songs,frozen,controls,evidence)


if __name__=='__main__':print(json.dumps(run(),indent=1))
